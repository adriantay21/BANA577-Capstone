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
import gradio as gr

load_dotenv()

openai.api_key = os.getenv("gpt_api_key")

storage_context = StorageContext.from_defaults(persist_dir=".\\IndexGPTCSV")

# Load index
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
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

chat_engine = index.as_chat_engine()

# Function to handle chat interaction
def chat(user_input, history):
    ai_response = chat_engine.chat(user_input)
    ai_text = ai_response.response  # Extract the text from AgentChatResponse
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

demo.launch()
