from langchain_openai import ChatOpenAI


class Model:
    def __init__(self, model, temperature):
        self.llm = ChatOpenAI(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio-local",
                model=model,
                temperature=temperature,
            )

    def getResponse(self, prompt):
        return self.llm.invoke(prompt)