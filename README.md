# localrag

`localrag` is a Python package enabling users to "chat" with their documents using a local Retrieval Augmented Generation (RAG) approach, without needing an external Large Language Model (LLM) provider.

It allows for quick, local, and easy interactions with text data, extracting and generating responses based on the content.

## Features

- **Local Processing**: Runs entirely on your local machine - no need to send data externally.
- **Customizable**: Easy to set up with default models or specify your own.
- **Versatile**: Use it for a variety of applications, from automated Q&A systems to data mining.

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
# can set device to mps or cuda:0 e.g device="mps"
# can set index location e.g index_location="my_index_loc"
my_local_rag = localrag.init()
# Add docs
my_local_rag.add_to_index("./docs")
# Chat with docs
response = my_local_rag.chat("What type of pet do I have?")
print(response.answer)
print(response.source_documents)
# Based on the context you provided, I can determine that you have a dog. Therefore, the type of pet you have is "dog."
# [Document(page_content='I have a dog', metadata={'source': 'docs/test.txt'})]
```

## License

This library is licensed under the Apache 2.0 License. See the LICENSE file.