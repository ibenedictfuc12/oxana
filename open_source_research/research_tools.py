import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import string
import time
import os

class ResearchTools:
    def __init__(self):
        self.architecture_list = ["resnet18", "resnet50", "vgg16", "mobilenet_v2"]

    def test_new_architecture(self, architecture_name, dataset_path):
        if architecture_name not in self.architecture_list:
            return f"{architecture_name} is not supported"
        torch.manual_seed(42)
        model = self._create_model(architecture_name)
        loader = self._create_dummy_loader(dataset_path)
        start_time = time.time()
        accuracy = self._train_and_evaluate(model, loader, epochs=1)
        elapsed = time.time() - start_time
        return f"Tested {architecture_name} on {dataset_path} with accuracy={accuracy:.2f}, time={elapsed:.2f}s"

    def compare_architectures(self, dataset_path):
        torch.manual_seed(42)
        results = {}
        for arch in self.architecture_list:
            model = self._create_model(arch)
            loader = self._create_dummy_loader(dataset_path)
            acc = self._train_and_evaluate(model, loader, epochs=1)
            results[arch] = acc
        best_arch = max(results, key=results.get)
        return f"Comparison results={results}, best_arch={best_arch}"

    def run_ablation_study(self, base_arch, dataset_path, remove_layers=None):
        if remove_layers is None:
            remove_layers = [False, True]
        results = []
        for remove in remove_layers:
            model = self._create_model(base_arch, remove_final_layer=remove)
            loader = self._create_dummy_loader(dataset_path)
            acc = self._train_and_evaluate(model, loader, epochs=1)
            mode = "with_full_model" if not remove else "with_removed_final_layer"
            results.append((mode, acc))
        return results

    def measure_performance(self, architecture_name, dataset_path, metric="loss"):
        model = self._create_model(architecture_name)
        loader = self._create_dummy_loader(dataset_path)
        torch.manual_seed(42)
        with torch.no_grad():
            batch = next(iter(loader))
            inputs, labels = batch
            outputs = model(inputs)
            if metric == "loss":
                loss = nn.functional.cross_entropy(outputs, labels).item()
                return f"Measured {metric}={loss:.4f} for {architecture_name}"
            elif metric == "inference_time":
                start = time.time()
                _ = model(inputs)
                diff = time.time() - start
                return f"Measured {metric}={diff:.4f}s for {architecture_name}"
        return "Unknown metric"

    def provide_educational_resources(self):
        return [
            "ArXiv: Generative Adversarial Networks",
            "ArXiv: Diffusion Models in Vision",
            "Tutorial: Sequence Modeling with Transformers"
        ]

    

    def synthetic_experiment(self, architecture_name, steps=10, input_shape=(1,3,224,224)):
        model = self._create_model(architecture_name)
        model.eval()
        torch.manual_seed(42)
        total_time = 0
        for _ in range(steps):
            x = torch.randn(input_shape)
            start = time.time()
            _ = model(x)
            total_time += time.time() - start
        avg_inference = total_time / steps
        return f"Synthetic test on {architecture_name} with average inference time={avg_inference:.4f}s"

   

    def save_experiment_checkpoint(self, model, checkpoint_dir="checkpoints"):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        filename = f"{self.generate_random_experiment_name()}.pt"
        path = os.path.join(checkpoint_dir, filename)
        torch.save(model.state_dict(), path)
        return path

    
    def _train_and_evaluate(self, model, loader, epochs=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for batch in loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0

    
        dataset = datasets.FakeData(transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader