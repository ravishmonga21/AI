import streamlit as st
from graph import app
from langchain_core.messages import HumanMessage
from streamlit_echarts import st_echarts

st.set_page_config(
    page_title="IMDB Movie Intelligence",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 IMDB Movie Intelligence")
st.caption("Ask anything about the top 10,000 IMDB movies")

# Chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("sql_response"):
            st.markdown(message["sql_response"])
        if message.get("chart_config"):
            st_echarts(options=message["chart_config"]["options"], height="400px")

# Chat input
if user_query := st.chat_input("Ask me about movies..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Run graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app.invoke({
                "messages": [HumanMessage(content=user_query)],
                "next": "supervisor",
                "user_query": user_query,
                "chart_config": None,
                "final_answer": ""
            })

        # Extract sql_agent response from messages
        sql_response = next(
            (msg.content for msg in reversed(result["messages"])
             if getattr(msg, "name", "") == "sql_agent"),
            None
        )

        chart_config = result.get("chart_config")

        # Always show SQL response
        if sql_response:
            st.markdown(sql_response)

        # Show chart only if viz was triggered
        if chart_config and chart_config.get("options"):
            st_echarts(options=chart_config["options"], height="400px")

    # Save to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "sql_response": sql_response,
        "chart_config": chart_config
    })