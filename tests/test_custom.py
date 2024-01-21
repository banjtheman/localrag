from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

import localrag

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load Chroma Vectordb
chroma_vectordb = Chroma(
    persist_directory="./chroma_db", embedding_function=embedding_function
)


# Custom embed text function
# the Vectorstore and the documents will be passed in
def chroma_add_docs(vectorstore, texts):
    print(texts)
    vectorstore.add_documents(texts)
    print("Added to vector store")


# Set up with all the necessary configurations
my_local_rag = localrag.custom_init(
    llm=llm,
    embedding_model=embedding_function,
    vectorstore=chroma_vectordb,
    custom_embed_text_func=chroma_add_docs,
    chunk_size=2000,
    chunk_overlap=40,
)

# Add a file
my_local_rag.add_to_index("pizza.txt")
response = my_local_rag.chat("What type of food do I like?")
print(response.answer)
print(response.context)
