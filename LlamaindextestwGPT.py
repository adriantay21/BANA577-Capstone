from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, get_response_synthesizer
from llama_index.core.schema import MetadataMode
from llama_index.core import VectorStoreIndex
import os
import openai
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage


load_dotenv()

openai.api_key = os.getenv("gpt_api_key")


storage_context = StorageContext.from_defaults(persist_dir=".\\IndexGPTCSV")

# load index
index = load_index_from_storage(storage_context)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)


response = query_engine.query("How to track energy use?")


print(response)

