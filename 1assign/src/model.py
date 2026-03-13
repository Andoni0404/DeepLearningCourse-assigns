import torch.nn as nn

# 1. Red Superficial (Shallow NN) - Falla por Underfitting (Bias Alto)
class ShallowInsuranceModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x): 
        return self.stack(x)

# 2. Red Profunda Optimizada (Ultimate Deep NN) - Modelo Ganador
class UltimateInsuranceModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2), # Regularización
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): 
        return self.stack(x)