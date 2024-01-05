from .chatresponse import ChatResponse
from .ragchat import RagChat

# Create a single instance of RagChat to be used for the module-level functions
chat_instance = None


def setup(
    llm_model: str = "llama2",
    embedding_model: str = "BAAI/bge-small-en",
    device: str = "mps",
    index_location: str = "localrag_index",
):
    """
    Setup the RAG Chat system with the specified models and configurations.
    """
    global chat_instance
    chat_instance = RagChat()
    chat_instance.setup(llm_model, embedding_model, device, index_location)


def chat(docs_path: str, query: str):
    """
    Simulate a chat with the document at the given path using the specified query.
    """
    global chat_instance
    if chat_instance is None:
        raise Exception("RagChat has not been set up. Please call setup() first.")
    response = chat_instance.chat(docs_path, query)
    return ChatResponse(response)
