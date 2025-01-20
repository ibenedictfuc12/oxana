import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import random
import os
import time

class ContentStyleNetwork(nn.Module):
    def __init__(self, content_layers=None, style_layers=None):
        super().__init__()
        if content_layers is None:
            content_layers = ["conv_4"]
        if style_layers is None:
            style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.vgg = models.vgg19(pretrained=True).features

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        content_features = {}
        style_features = {}
        layer_mapping = {
            '0': 'conv_1',
            '5': 'conv_2',
            '10': 'conv_3',
            '19': 'conv_4',
            '28': 'conv_5',
        }
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layer_mapping:
                if layer_mapping[name] in self.content_layers:
                    content_features[layer_mapping[name]] = x
                if layer_mapping[name] in self.style_layers:
                    style_features[layer_mapping[name]] = x
        return content_features, style_features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (c * h * w)

def compute_content_loss(gen_content, target_content):
    return nn.functional.mse_loss(gen_content, target_content)

def compute_style_loss(gen_style, target_style):
    G_gen = gram_matrix(gen_style)
    G_target = gram_matrix(target_style)
    return nn.functional.mse_loss(G_gen, G_target)

def run_style_transfer(content, style, content_weight=1e4, style_weight=1e2, steps=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContentStyleNetwork().to(device)

    content = content.to(device)
    style = style.to(device)

    target = content.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.01)

    content_features_c, style_features_c = model(content)
    content_features_s, style_features_s = model(style)

    for step in range(steps):
        optimizer.zero_grad()
        content_features_t, style_features_t = model(target)

        content_loss = 0.0
        style_loss = 0.0

        for layer in content_features_c:
            content_loss += compute_content_loss(
                content_features_t[layer], content_features_c[layer]
            )

        for layer in style_features_s:
            style_loss += compute_style_loss(
                style_features_t[layer], style_features_s[layer]
            )

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        optimizer.step()

    return target.detach().cpu()

def load_image(image_path, size=256):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

def save_image(tensor, path):
    transform = transforms.ToPILImage()
    image = transform(tensor.squeeze(0).cpu().clamp_(0, 1))
    image.save(path)

class StyleTransferAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def stylize_image(self, source_image, style_reference, output_path="stylized_output.jpg"):
        source = self._prepare_image(source_image)
        style = self._prepare_image(style_reference)
        start = time.time()
        stylized = run_style_transfer(source, style, steps=10)
        end = time.time()
        self._save_result(stylized, output_path)
        return f"Stylized image saved to '{output_path}' (elapsed={end - start:.2f}s)"

    def stylize_video(self, source_video, style_reference, output_dir="stylized_frames"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        style = self._prepare_image(style_reference)

        # Imitation: for a real pipeline, you'd extract frames from a video
        # Here we just create N dummy frames as an example
        num_frames = 5
        results = []

        for i in range(num_frames):
            # Fake a "frame" by loading a single source image or random noise
            frame_source = torch.randn(1, 3, 256, 256)
            frame_source = frame_source.to(self.device)
            stylized = run_style_transfer(frame_source, style, steps=5)
            frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
            self._save_result(stylized, frame_path)
            results.append(frame_path)

        return f"Stylized {num_frames} frames, saved to '{output_dir}'"

    def stylize_batch(self, image_paths, style_path, output_dir="batch_stylized"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        style = self._prepare_image(style_path)
        results = []

        for image_file in image_paths:
            source = self._prepare_image(image_file)
            stylized = run_style_transfer(source, style, steps=10)
            output_path = os.path.join(
                output_dir, f"stylized_{os.path.basename(image_file)}"
            )
            self._save_result(stylized, output_path)
            results.append(output_path)

        return f"Batch stylization completed, results in '{output_dir}'"

    def _prepare_image(self, image_path, size=256):
        if isinstance(image_path, str):
            return load_image(image_path, size=size).to(self.device)
        if isinstance(image_path, torch.Tensor):
            return image_path.to(self.device)
        return torch.randn(1, 3, size, size).to(self.device)

    def _save_result(self, tensor, path):
        save_image(tensor, path)