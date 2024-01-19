from .chatresponse import ChatResponse
from .ragchat import RagChat
from .utils import get_device_type


def init(
    llm_model="llama2",
    embedding_model="BAAI/bge-small-en-v1.5",
    device=get_device_type(),
    index_location="localrag_index",
    system_prompt=None,
):
    """
    Initialize a new instance of the RagChat system with specified or default configurations.

    Parameters:
    llm_model (str): The name of the large language model to use. Defaults to 'llama2'.
    embedding_model (str): The name of the embedding model to use. Defaults to 'BAAI/bge-small-en'.
    device (str): The device to run the models on. Defaults to 'cpu'.
    index_location (str): The location of the pre-built index for document retrieval. Defaults to 'localrag_index'.
    system_prompt (str): A system prompt for the model

    Returns:
    RagChat: A new instance of the RagChat class.
    """
    return RagChat(llm_model, embedding_model, device, index_location, system_prompt)
