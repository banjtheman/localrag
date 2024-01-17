import localrag

# Set up with all the necessary configurations
my_local_rag = localrag.init()

# Add a file
my_local_rag.add_to_index("pizza.txt")
response = my_local_rag.chat("What type of food do I like?")
print(response.answer)
print(response.context)

# Save Chat history
my_local_rag.save_chat_history("chat_history_test", "test")

# Init a new localrag
my_local_rag = localrag.init()

# Load the chat history
my_local_rag.load_chat_history("chat_history_test/test_history.pkl")

# Ask a follow up question
response = my_local_rag.chat("What are some popular toppings?")
print(response.answer)
print(response.context)
