from dotenv import load_dotenv
import os
from transformers import AutoTokenizer
import gradio as gr
from llama_index.llms.huggingface import HuggingFaceLLM, HuggingFaceInferenceAPI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, get_response_synthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import StorageContext, load_index_from_storage

load_dotenv()

hf_token = os.getenv("hf_token")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=hf_token,
)

# Define stopping IDs
stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids(""),
]

# Load LLM
llm = HuggingFaceInferenceAPI(
    model_name="https://htshuyazde0bg6dy.us-east-1.aws.endpoints.huggingface.cloud",
    token=hf_token
)

# Load embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Configure settings
Settings.embed_model = embed_model
Settings.llm = llm

# Load storage context and index
storage_context = StorageContext.from_defaults(persist_dir=".\\IndexLlamaCSV")
index = load_index_from_storage(storage_context)

# Configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# Configure response synthesizer
response_synthesizer = get_response_synthesizer()

# Assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

# Function to handle chat interaction
def chat(user_input, history):
    ai_response = query_engine.query(user_input)
    ai_text = ai_response.response  # Extract the text from the response
    history.append((user_input, ai_text))
    return history, history

# Create the Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    message = gr.Textbox(label="You:")
    state = gr.State([])

    def submit_message(user_input, history):
        new_history, updated_history = chat(user_input, history)
        return "", updated_history

    message.submit(submit_message, [message, state], [message, chatbot])

demo.launch(server_name="127.0.0.2", server_port=7861)
