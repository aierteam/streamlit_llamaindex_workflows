import streamlit as st
import os
import asyncio
from WikipediaRAGWorkflow import get_workflow

st.set_page_config(
    page_title="RAG Agent - Powered by LlamaIndex Workflows",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("📚 RAG Agent")
st.markdown("**Powered by LlamaIndex Workflows & OpenAI**")

# Sidebar
with st.sidebar:
    st.markdown(
        "Upload files (PDF, TXT, DOCX) and ask questions about them. Max 5MB per file."
    )
    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )
    st.markdown("💡 **Try asking:**")
    st.markdown(
        "- 'What is a transformer, and how is it used in large language models?'"
    )
    st.markdown("- 'How does supervised learning differ from unsupervised learning?'")
    st.markdown("- 'Describe the ethical concerns around large language models.'")

# file uploading
os.makedirs("data", exist_ok=True)

if uploaded:
    for fu in uploaded:
        if fu.size > 5 * 1024 * 1024:
            st.error(f"‘{fu.name}’ is too big ({fu.size/1024**2:.1f} MB). Max is 5 MB.")
        else:
            out_path = os.path.join("data", fu.name)
            with open(out_path, "wb") as f:
                f.write(fu.getbuffer())
    st.success(f"Saved {len(uploaded)} files to data.")


# Initialize the workflow agent
@st.cache_resource(show_spinner=False)
def get_agent():
    return get_workflow(api_key=st.secrets.openai_key)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your RAG agent. Ask me anything about your documents.",
        }
    ]

if "agent" not in st.session_state:
    st.session_state.agent = get_agent()
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload files or ask me anything."}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

        # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            # Run the workflow
            async def run_workflow():
                response_content = ""
                result = await st.session_state.agent.run(
                    query=prompt, dirname="data", streaming=True
                )

                if type(result) == str:
                    response_content = result
                else:
                    async for chunk in result.async_response_gen():
                        response_content += chunk
                        message_placeholder.write(response_content + "▌")

                return response_content

            # Run the async workflow
            final_response = asyncio.run(run_workflow())

            # if final_response == "Empty Response":
            #   llm = OpenAI(model="gpt-4o")
            #   final_response = llm.chat([
            #       ChatMessage(role="system", content="You are a helpful assistant."),
            #       ChatMessage(role="user", content=prompt)
            #   ]).message.content

            # Display final response
            message_placeholder.write(final_response)

            # Add to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            message_placeholder.write(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
    "Built with LlamaIndex Workflows • RAG Agent • OpenAI GPT-4o-mini"
    "</div>",
    unsafe_allow_html=True,
)
