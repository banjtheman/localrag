import localrag

# Set up with all the necessary configurations
my_local_rag = localrag.init()

# Add a file
my_local_rag.add_to_index("pizza.txt")
response = my_local_rag.chat("What type of food do I like?")
print(response.answer)
print(response.context)

print("Updating model!")
# Update model
my_local_rag.update_model("mixtral")
my_local_rag.clear_chat_history()
response = my_local_rag.chat("What type of food do I like?")
print(response.answer)
print(response.context)
