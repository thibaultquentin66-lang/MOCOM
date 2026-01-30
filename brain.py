import torch
import torch.nn as nn

class TrafficBrain(nn.Module):
    def __init__(self):
        super(TrafficBrain, self).__init__()
        # 4 Inputs (Queues N, S, E, W) -> 32 Hidden Neurons
        self.layer1 = nn.Linear(4, 32)
        # 32 Hidden -> 2 Outputs (Probabilities for Phase 0 or Phase 2)
        self.layer2 = nn.Linear(32, 2)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return torch.softmax(x, dim=0)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))