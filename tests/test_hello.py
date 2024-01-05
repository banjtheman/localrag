import localrag

# Set up with all the necessary configurations
localrag.setup()

# Now ready to chat!
response = localrag.chat("./docs", "What type of pet do I have?")
print(response)
print(response.source_documents)
