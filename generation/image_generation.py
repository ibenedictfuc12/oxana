import torch
import torch.nn as nn
import torch.optim as optim

class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
        )

class ImageGenerator:
    

    def generate_product_design(self, product_type, design_preference):
        z = torch.randn(1, self.latent_dim).to(self.device)
        generated = self.model(z).detach().cpu().numpy().tolist()
        return f"Generated product design for '{product_type}', preference='{design_preference}', data={generated}"

    def generate_nft_collection(self, collection_name, size):
        nft_items = []
        for i in range(size):
            z = torch.randn(1, self.latent_dim).to(self.device)
            data = self.model(z).detach().cpu().numpy().tolist()
            nft_items.append(f"{collection_name}_item_{i}_data={data}")
        return nft_items