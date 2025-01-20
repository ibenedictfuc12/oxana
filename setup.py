import os
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "torch>=1.9.0",
    "torchvision>=0.10.0",
    "torchaudio>=0.9.0",
    "numpy>=1.19.0",
    "scipy>=1.5.0",
    "Pillow>=8.0.0",
    "tqdm>=4.50.0",
    "requests>=2.25.0",
    "opencv-python>=4.5.0",
    "PyYAML>=5.3.0",
    "librosa>=0.8.0",
    "jinja2>=2.11.0",
    "scikit-learn>=0.24.0"
]

setup(
    name="ArtAI-Agent-Framework",
    version="0.1.0",
    author="ibenedictfuc12",
    description="A fictitious AI-based Art framework for generation, style transfer, and more.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ibenedictfuc12/oxana",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)