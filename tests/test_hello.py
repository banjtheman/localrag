import localrag

# Set up with all the necessary configurations
my_local_rag = localrag.init()

# Add a file
my_local_rag.add_to_index("pizza.txt")
response = my_local_rag.chat("What type of food do I like?")
print(response.answer)
print(response.context)

# Clear chat history
my_local_rag.clear_chat_history()

# Add a folder
my_local_rag.add_to_index("./docs")
response = my_local_rag.chat("What type of pet do I have?")
print(response.answer)
print(response.context)

# Clear chat history
my_local_rag.clear_chat_history()

# Add a website
my_local_rag.add_to_index("https://docs.smith.langchain.com/overview")
response = my_local_rag.chat("What does LangChain do?")
print(response.answer)
print(response.context)
