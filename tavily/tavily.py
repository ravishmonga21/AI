import sys
import json
import os
import re
import uuid
import subprocess
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain_tavily import TavilyExtract
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from llm_factory import get_llm

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

TAVILY_DIR = Path(__file__).resolve().parent
MAX_RETRIES = 3
llm = get_llm("claude-haiku-4-5")


# ── Step 1: Extract URLs (returns one entry per URL) ──

def extract_urls(urls: list[str]) -> list[dict]:
    """Returns [{"url": "...", "content": "..."}, ...] — one per input URL."""
    tool = TavilyExtract(extract_depth="basic", format="markdown", include_images=False)
    extracted = tool.invoke({"urls": urls})

    results = []
    if isinstance(extracted, dict) and "results" in extracted:
        for r in extracted["results"]:
            results.append({
                "url": r.get("url", ""),
                "content": r.get("raw_content") or r.get("content", ""),
            })
    else:
        for url in urls:
            results.append({"url": url, "content": str(extracted)})
    return results


# ── Step 2: Summarize → structured JSON ──

_summarizer_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=(TAVILY_DIR / "summaizer.yaml").read_text(),
)

def summarize(content: str) -> dict | None:
    response = _summarizer_agent.invoke({
        "messages": [HumanMessage(content=content[:15000])]
    })
    raw = response["messages"][-1].content.strip()
    raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("Summarizer returned invalid JSON, returning raw text")
        return None


# ── Step 3: Architecture Diagram Agent → Python code ──

def _fix_common_mistakes(code: str) -> str:
    """Patch known LLM mistakes before execution."""
    code = re.sub(
        r'Cluster\(([^)]*?),\s*style="([^"]*)"',
        r'Cluster(\1, graph_attr={"style": "\2"}',
        code,
    )
    return code


def generate_diagram_code(summary_json: dict, messages: list | None = None) -> str | None:
    """Generate diagram code. Pass prior messages for retry context."""
    arch_prompt = (TAVILY_DIR / "architecture.yaml").read_text()

    if messages is None:
        messages = [
            SystemMessage(content=arch_prompt),
            HumanMessage(content=json.dumps(summary_json, indent=2)),
        ]

    response = llm.invoke(messages)
    code = response.content.strip()
    code = re.sub(r"^```python|^```|```$", "", code, flags=re.MULTILINE).strip()

    if "from diagrams" not in code:
        print("Architecture agent did not produce valid diagrams code")
        return None

    code = _fix_common_mistakes(code)
    return code


# ── Step 4: Execute code → PNG (with ReAct retry) ──

def _slug_from_url(url: str) -> str:
    """Turn a URL into a short, filesystem-safe name for the PNG."""
    path = urlparse(url).path.rstrip("/").split("/")[-1]
    name = re.sub(r"\.html?$", "", path)
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name) or "diagram"


def _run_script(code: str, output_name: str, output_dir: str) -> dict:
    """Run generated code in a subprocess. Returns {success, png_path, stderr}."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    code = code.replace("__OUTPUT_PATH__", output_path)

    script_path = os.path.join(output_dir, f"_script_{output_name}.py")
    with open(script_path, "w") as f:
        f.write(code)

    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True, timeout=120, cwd=output_dir,
        )
        png_path = f"{output_path}.png"
        if result.returncode == 0 and os.path.exists(png_path):
            return {"success": True, "png_path": png_path, "stderr": ""}
        return {"success": False, "png_path": None, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"success": False, "png_path": None, "stderr": "Timed out (120s)"}
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


def execute_with_retry(
    summary_json: dict,
    output_name: str,
    output_dir: str = str(TAVILY_DIR / "output"),
) -> dict:
    """ReAct loop: generate code → run → if error, feed error back to LLM → retry."""
    arch_prompt = (TAVILY_DIR / "architecture.yaml").read_text()
    messages = [
        SystemMessage(content=arch_prompt),
        HumanMessage(content=json.dumps(summary_json, indent=2)),
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  Attempt {attempt}/{MAX_RETRIES}: generating code...")
        code = generate_diagram_code(summary_json, messages)
        if code is None:
            return {"png_path": None, "code": None, "error": "Failed to generate valid code"}

        messages.append(AIMessage(content=code))

        print(f"  Attempt {attempt}/{MAX_RETRIES}: executing...")
        run = _run_script(code, output_name, output_dir)

        if run["success"]:
            return {"png_path": run["png_path"], "code": code, "error": None}

        error_msg = run["stderr"][-2000:]  # truncate for token limits
        print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {error_msg[:200]}...")

        if attempt < MAX_RETRIES:
            messages.append(HumanMessage(content=(
                f"The code above failed with this error:\n\n{error_msg}\n\n"
                "Fix the code and return ONLY the corrected Python code. "
                "No explanations, no markdown fences."
            )))

    return {"png_path": None, "code": code, "error": f"Failed after {MAX_RETRIES} attempts"}


# ── Full Pipeline (one PNG per URL) ──

def run_pipeline(urls: list[str], output_dir: str = str(TAVILY_DIR / "output")) -> list[dict]:
    print(f"Step 1: Extracting {len(urls)} URL(s)...")
    pages = extract_urls(urls)

    results = []
    for i, page in enumerate(pages, 1):
        url = page["url"]
        content = page["content"]
        slug = _slug_from_url(url)
        entry = {"url": url, "png_path": None, "summary": None, "code": None, "error": None}

        print(f"\n[{i}/{len(pages)}] Processing: {url}")

        print("  Summarizing...")
        summary = summarize(content)
        if summary is None:
            entry["error"] = "Summarizer failed to produce valid JSON"
            results.append(entry)
            continue
        entry["summary"] = summary

        run = execute_with_retry(summary, slug, output_dir)
        entry["png_path"] = run["png_path"]
        entry["code"] = run["code"]
        entry["error"] = run["error"]

        results.append(entry)

    return results


# ── CLI entry point ──

if __name__ == "__main__":
    urls = [
        "https://enterprise-k8s.arcgis.com/en/latest/deploy/run-the-deployment-script.htm",
        "https://enterprise-k8s.arcgis.com/en/latest/create/example-end-to-end-deep-learning.htm",
        "https://enterprise-k8s.arcgis.com/en/11.4/introduction/what-is-arcgis-enterprise-kubernetes.htm",
        "https://www.esri.com/arcgis-blog/products/arcgis-enterprise/announcements/whats-new-in-arcgis-enterprise-12-0-on-kubernetes",
        "https://enterprise-k8s.arcgis.com/en/11.4/share/publish-web-tools.htm",
        "https://medium.com/@sverker_89371/arcgis-enterprise-on-kubernetes-a-peak-under-the-hood-e18cf6365d68",
    ]

    results = run_pipeline(urls)

    print("\n" + "=" * 60)
    for r in results:
        if r["png_path"]:
            components = len(r["summary"].get("components", []))
            print(f"  {r['png_path']}  ({components} components)")
        else:
            print(f"  FAILED: {r['url']} -- {r['error']}")
