# localrag

`localrag` is a Python package enabling users to "chat" with their documents using a local Retrieval Augmented Generation (RAG) approach, without needing an external Large Language Model (LLM) provider.

It allows for quick, local, and easy interactions with text data, extracting and generating responses based on the content.

## Features

- **Local Processing**: Runs entirely on your local machine - no need to send data externally.
- **Customizable**: Easy to set up with default models or specify your own.
- **Versatile**: Use it for a variety of applications, from automated Q&A systems to data mining. You add files, folders or websites to the index!

## Prerequisites

Before you install and start using `localrag`, make sure you meet the following requirements:

### Ollama for Local Inference

`localrag` uses Ollama for local inference, particularly beneficial for macOS users. Ollama allows for easy model serving and inference. To set up Ollama:

* [Download and run the app](https://ollama.ai/download)
* From command line, fetch a model from this [list of options](https://github.com/jmorganca/ollama): e.g., `ollama pull llama2`
* When the app is running, all models are automatically served on localhost:11434

## Installation

To install `localrag`, simply use pip:

```bash
pip install localrag
```

## Quick Start

Here's a quick example of how you can use localrag to chat with your documents:

Here is an example in test.txt in the docs folder:

```
I have a dog
```

```python
import localrag
my_local_rag = localrag.init()
# Add docs
my_local_rag.add_to_index("./docs")
# Chat with docs
response = my_local_rag.chat("What type of pet do I have?")
print(response.answer)
print(response.context)
# Based on the context you provided, I can determine that you have a dog. Therefore, the type of pet you have is "dog."
# [Document(page_content='I have a dog', metadata={'source': 'docs/test.txt'})]
```

### Website Example

```python
import localrag
my_local_rag = localrag.init()
my_local_rag.add_to_index("https://github.com/banjtheman/localrag")
response = my_local_rag.chat("What is localrag?")
print(response.answer)
# Based on the context provided in the GitHub repository page for "banjtheman/localrag", localrag is a chat application that allows users to communicate with their documents locally...
```

More examples in the [tests](./tests) folder.

## localrag config options

Here is how you can configure localrag:

```python
import localrag
my_local_rag = localrag.init(
    llm_model="llama2", # Can choose from ollama models: https://ollama.ai/library
    embedding_model="BAAI/bge-small-en-v1.5", # Can choose variations of https://huggingface.co/BAAI/bge-large-en-v1.5, top 5 embedding model https://huggingface.co/spaces/mteb/leaderboard
    device="mps", # can set device to mps, cpu or cuda:X
    index_location="localrag_index", # Location of the vectorstore
    system_prompt="You are Duck. Start each response with Quack.", # Custom system prompt
)
my_local_rag.add_to_index("./docs")

# can change the URL of the ollama server with my_local_rag.llm.base_url = "http://ollama:11434"
```

### localrag custom everything

You can provide the foloowing custom langchain objects:  
* llm
* vector databases (must also add an "add docs" function)
* embedding function

```python
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
    custom_embed_text_func=chroma_embed_text,
)

# Add a file
my_local_rag.add_to_index("pizza.txt")
response = my_local_rag.chat("What type of food do I like?")
print(response.answer)
print(response.context)
```

## License

This library is licensed under the Apache 2.0 License. See the LICENSE file.
