from setuptools import setup, find_packages

setup(
    name="bge-lora-fairness-finetuning",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fine-tuning BAAI/bge-large-en-v1.5 model with LoRA and fairness-aware techniques.",
    packages=find_packages(include=["src", "configs"]),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "datasets>=1.0.0",
        "scikit-learn>=0.24.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.50.0"
    ],
    entry_points={
        "console_scripts": [
            "train=bge-lora-fairness-finetuning.scripts.train:main",
            "evaluate=bge-lora-fairness-finetuning.scripts.evaluate:main",
            "inference=bge-lora-fairness-finetuning.scripts.inference:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)