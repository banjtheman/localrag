import localrag

# Set up with all the necessary configurations
my_local_rag = localrag.init()

# Add a file
my_local_rag.add_to_index("pizza.txt")
response = my_local_rag.chat("What type of food do I like?")
print(response.answer)
print(response.source_documents)

# Clear chat history
my_local_rag.clear_chat_history()

# Add a folder
my_local_rag.add_to_index("./docs")
response = my_local_rag.chat("What type of pet do I have?")
print(response.answer)
print(response.source_documents)
