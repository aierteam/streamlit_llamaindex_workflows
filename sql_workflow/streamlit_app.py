import streamlit as st
import os
import sys
import asyncio

# Add current directory to Python path to ensure local imports work
sys.path.append(os.path.dirname(__file__))

from sql_workflow import build_sql_workflow, SqlWorkflowStartEvent

st.set_page_config(
    page_title="SQL Agent - Powered by LlamaIndex Workflows",
    page_icon="🗃️",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("🗃️ SQL Database Agent")
st.markdown("**Powered by LlamaIndex Workflows & OpenAI**")

# Sidebar
with st.sidebar:
    st.markdown(
        "Ask natural language questions about the AdventureWorks database. The agent will convert your questions to SQL and execute them."
    )

    st.markdown("📊 **Available Tables:**")
    st.markdown("- **Customer**: Customer information (ID, name, email, etc.)")
    st.markdown(
        "- **SalesOrderHeader**: Sales order headers (order date, customer, total)"
    )
    st.markdown(
        "- **SalesOrderDetail**: Sales order details (products, quantities, prices)"
    )

    st.markdown("💡 **Try asking:**")
    st.markdown("- 'What is the total order amount for each customer?'")
    st.markdown("- 'Show me the top 10 customers by total sales'")
    st.markdown("- 'Which products have been ordered the most?'")
    st.markdown("- 'What were the sales trends by month?'")


# Initialize the workflow agent
@st.cache_resource(show_spinner=False)
def get_agent():
    try:
        return build_sql_workflow()
    except Exception as e:
        st.error(f"Failed to initialize SQL workflow: {str(e)}")
        return None


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your SQL database agent. Ask me questions about the AdventureWorks database and I'll convert them to SQL queries and provide answers.",
        }
    ]

if "agent" not in st.session_state:
    with st.spinner("Initializing SQL workflow..."):
        st.session_state.agent = get_agent()

    if st.session_state.agent is None:
        st.error(
            "Failed to initialize the SQL workflow. Please check your database connection."
        )
        st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Show SQL query if available in the message
        if message["role"] == "assistant" and "sql_query" in message:
            with st.expander("🔍 Generated SQL Query"):
                st.code(message["sql_query"], language="sql")

# Chat input
if prompt := st.chat_input("Ask me about the database..."):
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
                with st.spinner("Generating SQL and querying database..."):
                    start_event = SqlWorkflowStartEvent(query=prompt)
                    result = await st.session_state.agent.run(start_event=start_event)
                    return result

            # Run the async workflow
            result = asyncio.run(run_workflow())

            # Display the response
            message_placeholder.write(result.response)

            # Prepare message data for history
            message_data = {"role": "assistant", "content": result.response}

            # Show SQL query if available
            if result.reference and result.reference.sql_query:
                message_data["sql_query"] = result.reference.sql_query
                with st.expander("🔍 Generated SQL Query"):
                    st.code(result.reference.sql_query, language="sql")

                # Show database info if available
                if result.reference.sql_database_name:
                    st.caption(f"📊 Database: {result.reference.sql_database_name}")

            # Add to chat history
            st.session_state.messages.append(message_data)

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            message_placeholder.write(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

            # Show error details in expander
            with st.expander("🐛 Error Details"):
                st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
    "Built with LlamaIndex Workflows • SQL Agent • OpenAI GPT-4o"
    "</div>",
    unsafe_allow_html=True,
)

# Connection status indicator
with st.sidebar:
    st.markdown("---")
    if st.session_state.agent:
        st.success("✅ Database Connected")
    else:
        st.error("❌ Database Connection Failed")
