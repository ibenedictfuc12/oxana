![Oxana Banner](./img/oxana.png)

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/powered%20by-PyTorch-orange.svg)](https://pytorch.org/)

# oxana

A fictitious, yet comprehensive AI-based art framework designed for content generation, style transfer, image enhancement, and more.

**Repository**: [https://github.com/ibenedictfuc12/oxana](https://github.com/ibenedictfuc12/oxana)

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Tests](#tests)
- [Docker](#docker)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

**oxana** is a demo AI framework aimed at creative tasks such as image generation, style transfer, music and text generation, and more. While the code is illustrative and not fully production-ready, it demonstrates how an AI-based art engine could be structured and integrated into a larger ecosystem.

---

## Key Features

1. **Content Generation**  
   - Image generation (concept art, product design, NFT collections).  
   - Music composition and sound effects.  
   - Text generation (poems, short stories, brainstorming).  

2. **Style Transfer**  
   - Image and video stylization using neural style transfer.  

3. **Image Enhancement**  
   - Noise removal, detail restoration, and super-resolution.  

4. **AR/VR**  
   - Interactive installations with dynamic real-time content.  

5. **Analysis and Classification**  
   - Style recognition, forgery detection.  

6. **Educational Features**  
   - Virtual art teacher that provides lessons and examples.  

7. **Personalization**  
   - Custom decor generation, avatar creation.  

8. **Co-creation**  
   - Color palette suggestions, layout generation, design assistance.  

9. **Marketing**  
   - Ad material generation, brand identity preservation.  

10. **Open Source Research**  
   - Architecture testing, ablation studies, benchmarking.

---

## Installation

**Clone the repo**:
   ```bash
   git clone https://github.com/ibenedictfuc12/oxana.git
   cd oxana
   ```

**Install dependencies**:
```bash
pip install -e .
```

**(Optional) Install additional dependencies for testing**:
```bash
pip install pytest
```

---

## Usage

**Image Generation**:
```bash
python examples/generate_image.py --epochs 2
```

**Style Transfer**:
```bash
python examples/style_transfer_example.py --source input.jpg --style style.jpg
```

**Image Enhancement**:
```bash
python examples/image_enhancement_example.py --file sample.jpg --scale 2
```

---

## Configuration

-The config.yaml file allows centralized management of environment variables and hyperparameters. Example:
```bash
environment: development
framework:
  device: "cuda"
  logging_level: "INFO"

paths:
  data_dir: "./data"
  logs_dir: "./logs"
  checkpoints_dir: "./checkpoints"

training:
  batch_size: 8
  epochs: 10
  learning_rate: 0.001

style_transfer:
  content_weight: 1e4
  style_weight: 1e2
  steps: 50
  ```
-In the code, you can read these values as follows:
```bash
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = config["framework"]["device"]
epochs = config["training"]["epochs"]
```
