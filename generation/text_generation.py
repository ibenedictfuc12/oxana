import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleTextGenerator(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        h, _ = self.lstm(embedded)
        return self.fc(h)

class TextGenerator:
    def __init__(self):
        self.model = SimpleTextGenerator()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train_dummy(self, epochs=1):
        for _ in range(epochs):
            x = torch.randint(0, 1000, (2, 10)).to(self.device)
            y = torch.randint(0, 1000, (2, 10)).to(self.device)
            y_pred = self.model(x)
            loss = torch.mean((y_pred - nn.functional.one_hot(y, 1000).float()) ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def write_poem(self, topic, style):
        prompt = torch.randint(0, 1000, (1, 5)).to(self.device)
        output = self.model(prompt).argmax(dim=2).detach().cpu().numpy().tolist()
        return f"Poem for topic='{topic}', style='{style}', tokens={output}"

    def brainstorm_ideas(self, theme, format_type):
        ideas = []
        for _ in range(3):
            prompt = torch.randint(0, 1000, (1, 5)).to(self.device)
            out = self.model(prompt).argmax(dim=2).detach().cpu().numpy().tolist()
            ideas.append(f"Idea tokens={out}")
        return ideas