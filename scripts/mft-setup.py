from setuptools import setup, find_packages

setup(
    name="mistral-finetune",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "fire",
        "simple-parsing", 
        "pyyaml",
        "mistral-common>=1.3.1",
        "safetensors",
        "tensorboard", 
        "tqdm",
        "torch==2.2",
        "triton==2.2", 
        "xformers==0.0.24",
        "wandb"
    ]
)