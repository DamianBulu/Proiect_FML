from Model import Model


class AnalyserModel(Model):
    def __init__(self, model="lmstudio-community/meta-llama-3.1-8b-instruct", temperature=0.1):
        super().__init__(model, temperature)