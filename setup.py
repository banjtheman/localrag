from setuptools import find_packages, setup

setup(
    name="localrag",
    version="0.1.41",
    packages=find_packages(),
    install_requires=[
        "unstructured",
        "unstructured[all-docs]",
        "langchain",
        "langchain-openai",
        "langchain-community",
        "langchain-core",
        "faiss-cpu",
        "transformers",
        "sentence_transformers",
    ],
    # PyPI metadata
    author="Banjo Obayomi",
    author_email="banjtheman@gmail.com",
    description="Chat with your documents locally.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/banjtheman/localrag",
)
