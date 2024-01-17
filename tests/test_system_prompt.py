import localrag

# Set up with all the necessary configurations
my_local_rag = localrag.init(
    system_prompt="You are Duck. Start each response with Quack."
)

# Add a file
my_local_rag.add_to_index("pizza.txt")
response = my_local_rag.chat("What type of food do I like?")
print(response.answer)
print(response.context)

# Clear chat history
my_local_rag.clear_chat_history()


# Set up with all the necessary configurations
my_local_rag = localrag.init(
    system_prompt="You hate pizza. If someone likes it tell them they should consider trying chicken"
)

# Add a file
my_local_rag.add_to_index("pizza.txt")
response = my_local_rag.chat("What type of food do I like?")
print(response.answer)
print(response.context)

# Clear chat history
my_local_rag.clear_chat_history()
