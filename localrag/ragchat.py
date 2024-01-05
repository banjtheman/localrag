import os

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

from .chatresponse import ChatResponse


class RagChat:
    def __init__(
        self,
        llm_model="llama2",
        embedding_model="BAAI/bge-small-en",
        device="mps",
        index_location="localrag_index",
    ):
        """
        Initialize a new RagChat instance with specified or default configurations.

        Args:
            llm_model (str): The name of the large language model to use. Defaults to 'llama2'.
            embedding_model (str): The name of the embedding model to use. Defaults to 'BAAI/bge-small-en'.
            device (str): The device to run the models on. Defaults to 'mps'.
            index_location (str): The location of the pre-built index for document retrieval. Defaults to 'localrag_index'.
        """
        self.chat_history = []
        self.embeddings = None
        self.chain = None
        self.setup(llm_model, embedding_model, device, index_location)

    def setup(
        self,
        llm_model: str,
        embedding_model: str,
        device: str,
        index_location: str,
    ):
        """
        Setup the model and other configurations necessary for the RagChat system.

        Args:
            llm_model (str): The name of the large language model to use.
            embedding_model (str): The name of the embedding model to use.
            device (str): The device to run the models on.
            index_location (str): The location of the pre-built index for document retrieval.
        """
        # Setup the model and other configurations
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.device = device
        self.index_location = index_location

        self.setup_embeddings()

    def setup_embeddings(self):
        """
        Set up the embedding model based on the specified configuration.

        This typically involves loading the model into the specified device and
        preparing any other model-specific configurations.
        """
        encode_kwargs = {"normalize_embeddings": True}
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs=encode_kwargs,
        )

    def setup_llm(self):
        llm = Ollama(model=self.llm_model)

        vectorstore = FAISS.load_local(self.index_location, self.embeddings)

        self.chain = ConversationalRetrievalChain.from_llm(
            llm, vectorstore.as_retriever(), return_source_documents=True
        )

    def get_llm_response(self, user_query):
        """Interact with the agent and store chat history. Return the response."""

        result = self.chain({"question": user_query, "chat_history": self.chat_history})

        self.chat_history.append(HumanMessage(content=user_query))
        self.chat_history.append(AIMessage(content="Assistant: " + result["answer"]))
        return result

    def chunk_docs_to_text(self, docs_loc: str):
        loader = DirectoryLoader(docs_loc, silent_errors=True)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20
        )
        texts = text_splitter.split_documents(docs)

        return texts

    def embed_text(self, texts, save_loc: str):
        docsearch = FAISS.from_documents(texts, self.embeddings)

        docsearch.save_local(save_loc)

    def chat(self, docs_path: str, query: str):
        """
        Simulate a chat with the document at the given path using the specified query.

        Args:
            docs_path (str): The path to the directory or file containing the documents to chat with.
            query (str): The user's query or question to ask the document.

        Returns:
            ChatResponse: An object containing the response details, including the answer, question, and source documents.
        """
        if not os.path.exists(self.index_location):
            texts = self.chunk_docs_to_text(docs_path)
            self.embed_text(texts, self.index_location)

        if self.chain is None:
            self.setup_llm()

        response = self.get_llm_response(query)
        return ChatResponse(response)
