from Model import Model


class JudgeModel(Model):
    def __init__(self, model = "lmstudio-community/phi-3.1-mini-4k-instruct", temperature=0.1):
        super().__init__(model, temperature)
