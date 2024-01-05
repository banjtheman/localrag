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
    Initialize the RAG Chat system with specified or default configurations.

    This function sets up the necessary models and configurations for the chat system
    to work. It should be called before the chat function.

    Parameters:
    llm_model (str): The name of the large language model to use. Defaults to 'llama2'.
    embedding_model (str): The name of the embedding model to use. Defaults to 'BAAI/bge-small-en'.
    device (str): The device to run the models on. Defaults to 'mps'. "cpu" and "gpu" can work
    index_location (str): The location of the pre-built index for document retrieval. Defaults to 'localrag_index'.

    Returns:
    None
    """
    global chat_instance
    chat_instance = RagChat()
    chat_instance.setup(llm_model, embedding_model, device, index_location)


def chat(docs_path: str, query: str):
    """
    Simulate a chat with the document at the given path using the specified query.

    This function allows the user to interact with their documents in a conversational manner.
    It requires the setup function to be called beforehand to initialize the necessary configurations.

    Parameters:
    docs_path (str): The path to the directory or file containing the documents to chat with.
    query (str): The user's query or question to ask the document.

    Returns:
    ChatResponse: An object containing the response details, including the answer, question, and source documents.

    Raises:
    Exception: If the RagChat system has not been set up prior to calling this function.
    """
    global chat_instance
    if chat_instance is None:
        raise Exception("RagChat has not been set up. Please call setup() first.")
    response = chat_instance.chat(docs_path, query)
    return ChatResponse(response)
