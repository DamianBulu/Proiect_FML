from Model import Model


class AdvisorModel(Model):
    def __init__(self, model="lmstudio-community/medgemma-4b-it-MLX-4bit", temperature=0.7):
        super().__init__(model, temperature)