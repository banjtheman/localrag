from .ragchat import RagChat
from .utils import get_device_type


def init(
    llm_model="llama2",
    embedding_model="BAAI/bge-small-en-v1.5",
    device=get_device_type(),
    index_location="localrag_index",
    system_prompt=None,
    chunk_size: int = 1000,
    chunk_overlap: int = 20,
):
    """
    Initialize a new instance of the RagChat system with specified or default configurations.

    Parameters:
    llm_model (str): The name of the large language model to use. Defaults to 'llama2'.
    embedding_model (str): The name of the embedding model to use. Defaults to 'BAAI/bge-small-en'.
    device (str): The device to run the models on. Defaults to 'cpu'.
    index_location (str): The location of the pre-built index for document retrieval. Defaults to 'localrag_index'.
    system_prompt (str): A system prompt for the model
    chunk_size (int): Custom chunk size for text. Defaults to 1000
    chunk_overlap (int): Custom chunk size for overlap. Defaults to 20.
    Returns:
    RagChat: A new instance of the RagChat class.
    """
    return RagChat(
        llm_model=llm_model,
        embedding_model=embedding_model,
        device=device,
        index_location=index_location,
        system_prompt=system_prompt,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def custom_init(
    llm=None,
    embedding_model=None,
    vectorstore=None,
    custom_embed_text_func=None,
    device=get_device_type(),
    index_location="localrag_index",
    system_prompt=None,
    chunk_size: int = 1000,
    chunk_overlap: int = 20,
):
    """
    Initialize a new custom instance of the RagChat system with specified or default configurations.

    Parameters:
    llm: The langchain llm object'.
    embedding_model: The langchain embedding model object.
    vectorstore: The langchain vectorstore object.
    device (str): The device to run the models on. Defaults to 'cpu'.
    index_location (str): The location of the pre-built index for document retrieval. Defaults to 'localrag_index'.
    system_prompt (str): A system prompt for the model

    Returns:
    RagChat: A new instance of the RagChat class.
    """

    custom_llm = False
    custom_embedding_model = False
    custom_vectorstore = False

    if llm:
        custom_llm = True

    if embedding_model:
        custom_embedding_model = True

    if vectorstore:
        custom_vectorstore = True

    new_rag_chat = RagChat(
        device=device,
        system_prompt=system_prompt,
        has_custom_llm=custom_llm,
        index_location=index_location,
        has_custom_embeds=custom_embedding_model,
        has_custom_vector=custom_vectorstore,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if llm:
        new_rag_chat.llm = llm

    if embedding_model:
        new_rag_chat.embeddings = embedding_model

    if vectorstore:
        new_rag_chat.vectorstore = vectorstore
        new_rag_chat.custom_embed_text_func = custom_embed_text_func

    return new_rag_chat
