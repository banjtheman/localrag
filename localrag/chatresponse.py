class ChatResponse:
    def __init__(self, verbose_response):
        self.question = verbose_response.get("question", "")
        self.chat_history = verbose_response.get("chat_history", [])
        self.answer = verbose_response.get("answer", "")
        self.context = verbose_response.get("context", [])

    def __str__(self):
        # Customize this to print the information as you'd like
        return self.answer
