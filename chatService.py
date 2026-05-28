from langchain_openai import ChatOpenAI

class ChatService:
    def __init__(self):
        self.llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio-local",
        model="lmstudio-community/medgemma-4b-it-MLX-4bit",
        temperature=0.7
        )

    def getResponse(self, message):
        response = self.llm.stream(message)
        return response