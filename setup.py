from setuptools import setup, find_packages

setup(
    name="torchoptim",
    version="0.1.0",
    description="Model-agnostic ML inference optimization framework",
    author="Joel Crouch",
    author_email="joelcrouch@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
    "torch>=2.1.2",
    "transformers>=4.45.0",  # Required for Llama 3.2
    "accelerate>=0.26.0",
    "numpy<2.0.0",           # CRITICAL: Force NumPy 1.x
    "sentencepiece",
    "pillow",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.10.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
        "serving": [
            "vllm>=0.2.6",
            "fastapi>=0.104.1",
            "uvicorn>=0.24.0",
        ],
    },
)
