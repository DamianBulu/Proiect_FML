from Model import Model


class JudgeModel(Model):
    def __init__(self, model="TheBloke/medalpaca-13B-GGUF", temperature=0.1):
        super().__init__(model, temperature)
