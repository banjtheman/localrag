import os
import pickle

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredFileLoader,
    WebBaseLoader,
)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .chatresponse import ChatResponse
from .utils import get_device_type


class RagChat:
    def __init__(
        self,
        llm_model="llama2",
        embedding_model="BAAI/bge-small-en-v1.5",
        device=get_device_type(),
        index_location="localrag_index",
        system_prompt=None,
        has_custom_llm=False,
        has_custom_embeds=False,
        has_custom_vector=False,
        custom_embed_text_func=None,
        chunk_size: int = 1000,
        chunk_overlap: int = 20,
    ):
        """
        Initialize a new RagChat instance with specified or default configurations.

        Args:
            llm_model (str): The name of the large language model to use. Defaults to 'llama2'.
            embedding_model (str): The name of the embedding model to use. Defaults to 'BAAI/bge-small-en'.
            device (str): The device to run the models on. Defaults to 'cpu'.
            index_location (str): The location of the pre-built index for document retrieval. Defaults to 'localrag_index'.
            has_custom_llm (bool) = Flag to check if using custom LLM. Defaults to False,
            has_custom_embeds (bool) = Flag to check if using custom embedding model. Defaults to False,
            has_custom_vector (bool)=Flag to check if using custom vector database. Defaults to False,
            custom_embed_text_func: Custom function to add docs to a custom vector database.
        """
        self.chat_history = []
        self.embeddings = None
        self.chain = None
        self.vectorstore = None
        self.llm = None
        self.custom_embed_text_func = custom_embed_text_func
        self.has_custom_llm = has_custom_llm
        self.has_custom_embeds = has_custom_embeds
        self.has_custom_vector = has_custom_vector
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.setup(
            llm_model,
            embedding_model,
            device,
            index_location,
            system_prompt,
        )

    def setup(
        self,
        llm_model: str,
        embedding_model: str,
        device: str,
        index_location: str,
        system_prompt: str,
    ):
        """
        Setup the model and other configurations necessary for the RagChat system.

        Args:
            llm_model (str): The name of the large language model to use.
            embedding_model (str): The name of the embedding model to use.
            device (str): The device to run the models on.
            index_location (str): The location of the pre-built index for document retrieval.
            system_prompt (str): A system prompt for the model
        """
        # Setup the model and other configurations
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.device = device
        self.index_location = index_location
        self.system_prompt = system_prompt

        if not self.has_custom_embeds:
            self.setup_embeddings()
        if not self.has_custom_vector:
            self.setup_vectorstore()
        if not self.has_custom_llm:
            self.setup_llm()

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
        """Create a new LLM"""
        self.llm = Ollama(model=self.llm_model)

    def update_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt

        # Update chain with new prompt
        self.setup_chain()

    def setup_chain(self):
        """Create a new LLM chain with the vectorstore"""

        if self.llm is None:
            self.setup_llm()

        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
                ),
            ]
        )

        retriever = self.vectorstore.as_retriever()

        retriever_chain = create_history_aware_retriever(self.llm, retriever, prompt)

        if self.system_prompt:
            prompt_string = (
                self.system_prompt
                + " Answer the user's questions based on the below context:\n\n{context}"
            )
        else:
            prompt_string = (
                "Answer the user's questions based on the below context:\n\n{context}"
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    prompt_string,
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        self.chain = create_retrieval_chain(retriever_chain, document_chain)

    def update_model(self, model):
        """Update the model"""
        try:
            self.llm_model = model
            self.llm = Ollama(model=model)

        except Exception as e:
            print(f"Model update failed {e}")

    def get_llm_response(self, user_query):
        """Interact with the agent and store chat history. Return the response."""

        result = self.chain.invoke(
            {"input": user_query, "chat_history": self.chat_history}
        )

        self.chat_history.append(HumanMessage(content=user_query))
        self.chat_history.append(AIMessage(content=result["answer"]))

        return result

    def chunk_docs_to_text(self, docs_loc: str):
        loader = DirectoryLoader(docs_loc, silent_errors=True)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(docs)

        return texts

    def chunk_doc_to_text(self, doc_loc: str):
        loader = UnstructuredFileLoader(doc_loc, silent_errors=True)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(docs)

        return texts

    def chunk_website_to_text(self, website_loc: str):
        loader = WebBaseLoader(website_loc)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(docs)

        return texts

    def embed_text(self, texts):
        docsearch = FAISS.from_documents(texts, self.embeddings)

        # Update vectorstore
        self.update_vectorstore(docsearch)

    def add_to_index(self, doc: str):
        """
        Add item to the vector index

        Args:
            doc (str): The path to the directory or file containing the documents to index.
        """
        try:
            if os.path.isdir(doc):
                texts = self.chunk_docs_to_text(doc)

                if self.custom_embed_text_func:
                    self.custom_embed_text_func(self.vectorstore, texts)
                else:
                    self.embed_text(texts)
            elif os.path.isfile(doc):
                texts = self.chunk_doc_to_text(doc)
                if self.custom_embed_text_func:
                    self.custom_embed_text_func(self.vectorstore, texts)
                else:
                    self.embed_text(texts)
            else:
                # website?
                texts = self.chunk_website_to_text(doc)
                if self.custom_embed_text_func:
                    self.custom_embed_text_func(self.vectorstore, texts)
                else:
                    self.embed_text(texts)
        except Exception as e:
            print(f"Failed... {e}")

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
            self.setup_chain()

        response = self.get_llm_response(query)
        return ChatResponse(response)
