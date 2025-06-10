import streamlit as st
import asyncio
from WikipediaRAGWorkflow import get_workflow

st.set_page_config(
    page_title="Wikipedia RAG Agent",
    page_icon="📚",
    layout="centered",
)

st.title("📚 RAG Chat with Wikipedia")
st.markdown("Ask a question and get a concise answer powered by RAG.")

# Cache the workflow instance (with your OpenAI key from Streamlit secrets)
@st.cache_resource(show_spinner=False)
def init_wf():
    return get_workflow(api_key=st.secrets.openai_key)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! What Wikipedia topic can I help you with today?"}
    ]
if "workflow" not in st.session_state:
    st.session_state.workflow = init_wf()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

        with st.chat_message("assistant"):
          placeholder = st.empty()

          async def run_rag():
              text = ""
              wf = st.session_state.workflow
              run = wf.run(query=prompt)
              async for chunk in run.async_response_gen():
                  text += chunk
                  # stream interim result
                  placeholder.write(text + "▌")
              return text

          # execute and stream
          final_text = asyncio.run(run_rag())

          # write final
          placeholder.write(final_text)

          # save to history
          st.session_state.messages.append(
              {"role": "assistant", "content": final_text}
          )
