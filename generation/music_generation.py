import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleMusicGenerator(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h)

class MusicGenerator:
    def __init__(self):
        self.model = SimpleMusicGenerator()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train_dummy(self, epochs=1):
        for _ in range(epochs):
            x = torch.randn(4, 32, 16).to(self.device)
            y = torch.randn(4, 32, 16).to(self.device)
            y_pred = self.model(x)
            loss = torch.mean((y_pred - y) ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compose_track(self, genre, length):
        x = torch.randn(1, length, 16).to(self.device)
        output = self.model(x).detach().cpu().numpy().tolist()
        return f"Composed {genre} track with data={output}"

    def generate_sound_effects(self, effect_type, duration):
        x = torch.randn(1, duration, 16).to(self.device)
        output = self.model(x).detach().cpu().numpy().tolist()
        return f"Generated {effect_type} effect data={output}"