from dotenv import load_dotenv
import os
from transformers import AutoTokenizer
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.core.llms import ChatMessage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, get_response_synthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import StorageContext, load_index_from_storage

from transformers import AutoModel

load_dotenv()

hf_token = os.getenv("hf_token")



tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=hf_token,
)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]


llm = HuggingFaceInferenceAPI(
    model_name="https://htshuyazde0bg6dy.us-east-1.aws.endpoints.huggingface.cloud",
    token= hf_token
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# embed_model = AutoModel.from_pretrained('bert-base-uncased')

Settings.embed_model = embed_model

Settings.llm = llm


storage_context = StorageContext.from_defaults(persist_dir=".\\IndexLlamaCSV")

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
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0)],
)



response = query_engine.query("How to track energy use?")

print(response)