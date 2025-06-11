from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.workflow import Context, Workflow, Event, StartEvent, StopEvent, step

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

class IngestEvent(Event):
    index: VectorStoreIndex
    
class RetrieverEvent(Event):
    nodes: list[NodeWithScore]

class RerankEvent(Event):
    nodes: list[NodeWithScore]

class RAGWorkflow(Workflow):
    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> IngestEvent | None:
        dirname = ev.get("dirname")
        if not dirname:
            return None
        
        query = ev.get("query", None)
        if query is None:
            return None

        await ctx.set("query", query)

        documents = SimpleDirectoryReader(dirname).load_data()
        index = VectorStoreIndex.from_documents(
            documents=documents,
            embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
        )
        return IngestEvent(index=index)
    
    @step
    async def retrieve(self, ctx: Context, ev: IngestEvent) -> RetrieverEvent | None:
        index = ev.index
        query = await ctx.get("query", None)
        if (query is None) or (index is None):
            return None

        retriever = index.as_retriever(similarity_top_k=5)
        nodes = await retriever.aretrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        ranker = LLMRerank(
            choice_batch_size=5,
            top_n=3,
            llm=OpenAI(model="gpt-4o"),
        )
        query = await ctx.get("query", default=None)
        new_nodes = ranker.postprocess_nodes(ev.nodes, query_str=query)
        print(f"Reranked nodes to {len(new_nodes)}")
        return RerankEvent(nodes=new_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        llm = OpenAI(model="gpt-4o")
        summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
        query = await ctx.get("query", default=None)
        
        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)

def get_workflow(api_key: str) -> RAGWorkflow:
    import os
    os.environ["OPENAI_API_KEY"] = api_key
    return RAGWorkflow(timeout=60, verbose=True)