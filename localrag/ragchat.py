from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS


class RagChat:
    def __init__(self):
        # Initialize with default or empty settings
        self.llm_model = "llama2"
        self.embedding_model = "BAAI/bge-small-en"
        self.device = "mps"
        self.index_location = "localrag_index"
        self.chat_history = []
        self.embeddings = None
        self.chain = None

    def setup(
        self,
        llm_model: str = "llama2",
        embedding_model: str = "BAAI/bge-small-en",
        device: str = "mps",
        index_location: str = "localrag_index",
    ):
        # Setup the model and other configurations
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.device = device
        self.index_location = index_location

        self.setup_embeddings()

    def setup_embeddings(self):
        encode_kwargs = {"normalize_embeddings": True}
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs=encode_kwargs,
        )

    def setup_llm(self):
        # Set Context for response
        TEMPLATE = """You are expert AI assistant . Your role is to help user's chat with thier doucments. Return your response in markdown, so you can bold and highlight important information for users. If the answer cannot be found within the context, write 'I could not find an answer from the documents.' 

        Here is the conversation so far:
        {chat_history}

        Use the following context from the user's documents and the chat history to answer the user's query. Make sure to read all the context before providing an answer.\nContext:\n{context}\nQuestion: {question}
        """

        QA_PROMPT = PromptTemplate(
            template=TEMPLATE, input_variables=["question", "context"]
        )

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
        loader = DirectoryLoader(docs_loc)
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
        # This is where you implement the chat functionality
        # Integrate the previous methods to complete the chat operation
        # E.g., chunk docs, embed text, and then get LLM response

        texts = self.chunk_docs_to_text(docs_path)
        self.embed_text(texts, self.index_location)

        if self.chain is None:
            self.setup_llm()

        response = self.get_llm_response(query)

        return response
