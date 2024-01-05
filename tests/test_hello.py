import localrag

# Set up with all the necessary configurations
my_local_rag = localrag.init()

# Now ready to chat!
response = my_local_rag.chat("./docs", "What type of pet do I have?")
print(response.answer)
print(response.source_documents)
