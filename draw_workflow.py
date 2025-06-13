import os
import sys

from sql_workflow.sql_workflow import SimpleSqlDBWorkflow
from llama_index.utils.workflow import draw_all_possible_flows

# Create the workflow_diagrams directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), "workflow_diagrams")
os.makedirs(output_dir, exist_ok=True)

draw_all_possible_flows(
    SimpleSqlDBWorkflow,
    filename=os.path.join(output_dir, "simple_sql_db_workflow.html"),
)
