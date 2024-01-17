import os
import pickle

from langchain.chains import ConversationalRetrievalChain
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

from .chatresponse import ChatResponse


class RagChat:
    def __init__(
        self,
        llm_model="llama2",
        embedding_model="BAAI/bge-small-en",
        device="cpu",
        index_location="localrag_index",
    ):
        """
        Initialize a new RagChat instance with specified or default configurations.

        Args:
            llm_model (str): The name of the large language model to use. Defaults to 'llama2'.
            embedding_model (str): The name of the embedding model to use. Defaults to 'BAAI/bge-small-en'.
            device (str): The device to run the models on. Defaults to 'cpu'.
            index_location (str): The location of the pre-built index for document retrieval. Defaults to 'localrag_index'.
        """
        self.chat_history = []
        self.embeddings = None
        self.chain = None
        self.vectorstore = None
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
        self.setup_vectorstore()

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

    def setup_vectorstore(self):
        """Load a vectorstore if it exists"""
        if os.path.exists(self.index_location):
            self.vectorstore = FAISS.load_local(self.index_location, self.embeddings)

    def update_vectorstore(self, new_db):
        """Update or create a new vector store"""
        if self.vectorstore:
            self.vectorstore.merge_from(new_db)
            self.vectorstore.save_local(self.index_location)
        else:
            self.vectorstore = new_db
            self.vectorstore.save_local(self.index_location)

    def setup_llm(self):
        """Create a new LLM chain with the vectorstore"""
        llm = Ollama(model=self.llm_model)

        self.chain = ConversationalRetrievalChain.from_llm(
            llm, self.vectorstore.as_retriever(), return_source_documents=True
        )

    def update_model(self, model):
        """Update the model"""
        try:
            llm = Ollama(model=model)
            self.llm_model = model

            self.chain = ConversationalRetrievalChain.from_llm(
                llm, self.vectorstore.as_retriever(), return_source_documents=True
            )
        except Exception as e:
            print(f"Model update failed {e}")

    def get_llm_response(self, user_query):
        """Interact with the agent and store chat history. Return the response."""

        result = self.chain({"question": user_query, "chat_history": self.chat_history})

        self.chat_history.append(HumanMessage(content=user_query))
        self.chat_history.append(AIMessage(content="Assistant: " + result["answer"]))

        # name_query = "Given the interaction so far, how would name this chat? Only respond with the name of the chat, Im a python program not a human, I only need the name of the chat. Please keep the name breif"
        # name_result = self.chain(
        #     {"question": name_query, "chat_history": self.chat_history}
        # )
        # print("Chat name:")
        # print(name_result["answer"])

        return result

    def chunk_docs_to_text(self, docs_loc: str):
        loader = DirectoryLoader(docs_loc, silent_errors=True)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20
        )
        texts = text_splitter.split_documents(docs)

        return texts

    def chunk_doc_to_text(self, doc_loc: str):
        loader = UnstructuredFileLoader(doc_loc, silent_errors=True)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20
        )
        texts = text_splitter.split_documents(docs)

        return texts

    def embed_text(self, texts, save_loc: str):
        docsearch = FAISS.from_documents(texts, self.embeddings)

        # Update vectorstore
        self.update_vectorstore(docsearch)

    def add_to_index(self, doc: str):
        """
        Add item to the vector index

        Args:
            doc (str): The path to the directory or file containing the documents to index.
        """

        if os.path.isdir(doc):
            texts = self.chunk_docs_to_text(doc)
            self.embed_text(texts, self.index_location)
        elif os.path.isfile(doc):
            texts = self.chunk_doc_to_text(doc)
            self.embed_text(texts, self.index_location)

    def clear_chat_history(self):
        """
        Clear chat history
        """
        self.chat_history = []

    def save_chat_history(self, output_dir: str, chat_name: str):
        """
        Save chat history to a pickle file in the specified output directory.

        Args:
            output_dir (str): The directory where the chat history file will be saved.
            chat_name (str): Name of the chat, used for naming the pickle file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # Create the directory if it doesn't exist

        file_path = os.path.join(output_dir, f"{chat_name}_history.pkl")

        try:
            with open(file_path, "wb") as file:
                pickle.dump(self.chat_history, file)
            print(f"Chat history saved successfully to {file_path}.")
        except Exception as e:
            print(f"Error saving chat history: {e}")

    def load_chat_history(self, file_path: str):
        """
        Load chat history from a pickle file located at the given file path.

        Args:
            file_path (str): Full path to the pickle file containing the chat history.
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    self.chat_history = pickle.load(file)
                print(f"Chat history loaded successfully from {file_path}.")
            else:
                print(f"No chat history found at {file_path}.")
                self.chat_history = []  # Reset or initialize chat history
        except Exception as e:
            print(f"Error loading chat history: {e}")
            self.chat_history = (
                []
            )  # Reset or initialize chat history in case of an error

    def chat(self, query: str):
        """
        Chat with the current docs in the vector index.

        Args:
            query (str): The user's query or question to ask the document.

        Returns:
            ChatResponse: An object containing the response details, including the answer, question, and source documents.
        """
        if self.chain is None:
            self.setup_llm()

        response = self.get_llm_response(query)
        return ChatResponse(response)
