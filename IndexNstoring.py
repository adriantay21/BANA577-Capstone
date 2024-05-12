from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, get_response_synthesizer
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import openai
import os
from llama_index.readers.file import CSVReader, PandasCSVReader
load_dotenv()

openai.api_key = os.getenv("gpt_api_key")

# #embed for llama
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Settings.embed_model = embed_model



parser = PandasCSVReader()
file_extractor = {".csv": parser}  # Add other CSV formats as needed
documents = SimpleDirectoryReader(
    input_dir="C:\\Users\\adria\\OneDrive\\Desktop\\COBChatBot\\Data", file_extractor=file_extractor
).load_data()

# reader = SimpleDirectoryReader(input_dir="C:\\Users\\adria\\OneDrive\\Desktop\\COBChatBot\\Data")
# documents = reader.load_data()

# documents[0].excluded_llm_metadata_keys = []
print(len(documents))
# for x in range(len(documents)):
#     documents[x].excluded_llm_metadata_keys = []
#     documents[x].excluded_embed_metadata_keys = []
# breakpoint()



index = VectorStoreIndex.from_documents(documents,show_progress=True)

index.storage_context.persist(persist_dir="C:\\Users\\adria\\OneDrive\\Desktop\\COBChatBot\\Data\\IndexGPTCSV")