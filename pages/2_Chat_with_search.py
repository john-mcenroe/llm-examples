import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

# Sidebar for API key input
with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="openai_api_key", type="password"
    )
    "[Fetch an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("📝 File Q&A with OpenAI")

# File uploader and question input
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    else:
        article = uploaded_file.read().decode()
        prompt = f"Here's an article:\n\n{article}\n\n{question}"

        # Initialize the agent with OpenAI's model
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
        search = DuckDuckGoSearchRun(name="Search")
        search_agent = initialize_agent(
            [search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
        )

        # Use StreamlitCallbackHandler for interactive output
        with st.container():
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run([{"role": "user", "content": prompt}], callbacks=[st_cb])
            st.write("### Answer")
            st.write(response)
