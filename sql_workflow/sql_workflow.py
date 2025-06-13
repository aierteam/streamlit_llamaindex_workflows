from llama_index.core.workflow import (
    Workflow,
    step,
    StartEvent,
    StopEvent,
    Context,
    Event,
)
from llama_index.core.llms import LLM, ChatResponse
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.objects import (
    SQLTableSchema,
    SQLTableNodeMapping,
    ObjectRetriever,
    ObjectIndex,
)
from sqlalchemy import create_engine
import re
import tiktoken
import asyncio
from typing import List
from typing import Optional
from pydantic import BaseModel


# # === Reference Info Class ===
class ReferenceInfo(BaseModel):
    sql_query: Optional[str] = None
    sql_database_name: Optional[str] = None


# === Event Definitions ===
class SqlWorkflowStartEvent(StartEvent):
    query: str


class SQLEvent(Event):
    sql: str
    query: str


class CommonStopEvent(StopEvent):
    response: str
    reference: Optional[ReferenceInfo] = None


# === Utility Functions ===
def parse_response_to_sql(chat_response: ChatResponse) -> str:
    text = chat_response.message.content.strip()
    text = re.sub(r"```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```", "", text)
    if text.startswith("SQLQuery:"):
        text = text[len("SQLQuery:") :].strip()
    text = re.sub(r"--.*", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines).rstrip(";").strip()


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def get_table_context_str(
    sql_database: SQLDatabase, table_schema_objs: List[SQLTableSchema]
):
    context_strs = []
    for schema in table_schema_objs:
        info = sql_database.get_single_table_info(schema.table_name)
        if schema.context_str:
            info += " Description: " + schema.context_str
        context_strs.append(info)
    return "\n\n".join(context_strs)


# === Simple Two-Step SQL Workflow ===
class SimpleSqlDBWorkflow(Workflow):
    def __init__(
        self,
        sql_database: SQLDatabase,
        obj_retriever: ObjectRetriever,
        sql_retriever: SQLRetriever,
        text2sql_prompt: PromptTemplate,
        response_synthesis_prompt: PromptTemplate,
        llm: LLM = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sql_database = sql_database
        self.obj_retriever = obj_retriever
        self.sql_retriever = sql_retriever
        self.text2sql_prompt = text2sql_prompt
        self.llm = llm or OpenAI(model="gpt-4o-mini")
        self.response_synthesis_prompt = response_synthesis_prompt

    @step
    def generate_sql(self, ctx: Context, ev: SqlWorkflowStartEvent) -> SQLEvent:
        """Retrieve schema and generate SQL in one step"""
        # Get schema
        table_schemas = self.obj_retriever.retrieve(ev.query)
        schema_str = get_table_context_str(self.sql_database, table_schemas)

        # Generate SQL
        messages = self.text2sql_prompt.format_messages(
            query_str=ev.query, schema_description=schema_str
        )
        response = self.llm.chat(messages)
        sql = parse_response_to_sql(response)

        return SQLEvent(sql=sql, query=ev.query)

    @step
    def execute_sql(self, ctx: Context, ev: SQLEvent) -> CommonStopEvent:
        """Sanitize and execute SQL in one step"""
        # Simple sanitization - just add brackets to tables
        sql = ev.sql
        table_names = ["SalesOrderHeader", "SalesOrderDetail", "Customer"]
        for table_name in table_names:
            import re

            pattern = rf"(?<!\[)\b{table_name}\b(?!\])"
            sql = re.sub(pattern, f"[{table_name}]", sql)

        # Add schema prefix for reference (manual execution)
        full_sql = sql
        for table_name in table_names:
            pattern = rf"\[{table_name}\]"
            full_sql = re.sub(pattern, f"[Sales].[{table_name}]", full_sql)

        try:
            rows = self.sql_retriever.retrieve(sql)
            result_str = str(rows)

            if count_tokens(result_str) > 8000:
                return CommonStopEvent(
                    response="Result too long. Please run this SQL manually.",
                    reference=ReferenceInfo(
                        sql_query=full_sql,
                        sql_database_name=self.sql_database.engine.url.database,
                    ),
                )

            fmt_messages = self.response_synthesis_prompt.format_messages(
                sql_query=full_sql,
                context_str=result_str,
                query_str=ev.query,
            )
            response = self.llm.chat(fmt_messages)

            return CommonStopEvent(
                response=response.message.content,
                reference=ReferenceInfo(
                    sql_query=full_sql,
                    sql_database_name=self.sql_database.engine.url.database,
                ),
            )
        except Exception as e:
            return CommonStopEvent(
                response=f"SQL execution error: {e}",
                reference=ReferenceInfo(
                    sql_query=full_sql,
                    sql_database_name=self.sql_database.engine.url.database,
                ),
            )


# === Factory Function to Build Workflow ===
def build_sql_workflow() -> SimpleSqlDBWorkflow:
    # Connect to local SQL Server from Docker container
    # import os
    # db_host = os.getenv('SQL_SERVER_HOST', 'host.docker.internal')
    # db_user = os.getenv('SQL_SERVER_USER', '')
    # db_password = os.getenv('SQL_SERVER_PASSWORD', '')

    # if db_user and db_password:
    # Use SQL Server authentication
    import os

    # Try different connection methods for different environments
    try:
        # First try ODBC Driver 17 for SQL Server
        driver = "ODBC+Driver+17+for+SQL+Server"
        connection_string = (
            f"mssql+pyodbc://sa:AierTeam%402025@64.227.110.137:1433/AdventureWorksLT2022?"
            f"driver={driver}&TrustServerCertificate=yes"
        )
        # Test the connection
        engine = create_engine(connection_string)
        engine.connect().close()

    except Exception as e:
        # Fallback to ODBC Driver 18 for SQL Server
        print(f"ODBC Driver 17 connection failed: {e}")
        print("Trying ODBC Driver 18...")
        driver = "ODBC+Driver+18+for+SQL+Server"
        connection_string = (
            f"mssql+pyodbc://sa:AierTeam%402025@64.227.110.137:1433/AdventureWorksLT2022?"
            f"driver={driver}&TrustServerCertificate=yes"
        )
        engine = create_engine(connection_string)

    sql_database = SQLDatabase(
        engine,
        include_tables=["Customer", "SalesOrderHeader", "SalesOrderDetail"],
        schema="SalesLT",
    )

    table_infos_json = {
        "Customer": {
            "description": "Contains customer information such as CustomerID, NameStyle, CompanyName, EmailAddress, and demographics."
        },
        "SalesOrderHeader": {
            "description": "Represents the header of sales orders including SalesOrderID, OrderDate, CustomerID, TotalDue, and ShipMethod."
        },
        "SalesOrderDetail": {
            "description": "Contains details for each sales order item, including SalesOrderID, ProductID, OrderQty, UnitPrice, and LineTotal."
        },
    }

    table_schemas = [
        SQLTableSchema(table_name=name, context_str=info["description"])
        for name, info in table_infos_json.items()
    ]
    table_node_mapping = SQLTableNodeMapping(sql_database)
    obj_index = ObjectIndex.from_objects(table_schemas, table_node_mapping)
    obj_retriever = obj_index.as_retriever(similarity_top_k=1)

    sql_retriever = SQLRetriever(sql_database)

    # Create a custom prompt template for text-to-SQL
    custom_prompt = PromptTemplate(
        """Given the following database schema information:
        {schema_description}
        Please generate a SQL query to answer this question: {query_str}

        Instructions:
        - Use only the tables and columns shown in the schema above
        - Generate valid SQL syntax for Microsoft SQL Server
        - Use table names as shown (do not add schema prefixes)
        - Wrap table and column names in square brackets if needed for SQL Server
        - Return only the SQL query, no explanation
        - Use proper table aliases if needed

        SQL Query:"""
    )

    llm = OpenAI(model="gpt-4o")
    response_synthesis_prompt = PromptTemplate(
        """Given an input question, synthesize a response from the query results.

        Query: {query_str}
        SQL: {sql_query}
        SQL Response: {context_str}
        Response:"""
    )

    return SimpleSqlDBWorkflow(
        sql_database=sql_database,
        obj_retriever=obj_retriever,
        sql_retriever=sql_retriever,
        text2sql_prompt=custom_prompt,
        response_synthesis_prompt=response_synthesis_prompt,
        llm=llm,
    )


# # Simple workflow test
# async def test_workflow():
#     workflow = build_sql_workflow()
#     query = "What is the total order amount for each customer?"
#     print(f"\n🔍 Test Query: {query}")

#     try:
#         start_event = SqlWorkflowStartEvent(query=query)
#         result = await workflow.run(start_event=start_event)
#         print(f"✅ Success! Result:\n{result.response}")

#         # Display SQL query if available
#         if result.reference and result.reference.sql_query:
#             print(f"\n🔍 Generated SQL Query:\n{result.reference.sql_query}")
#         if result.reference and result.reference.sql_database_name:
#             print(f"\n📊 Database: {result.reference.sql_database_name}")

#     except Exception as e:
#         print(f"❌ Failed: {e}")


# # Run the test
# if __name__ == "__main__":
#     asyncio.run(test_workflow())
