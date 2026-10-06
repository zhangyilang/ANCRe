import setuptools

setuptools.setup(
    name="ancre",
    version="0.0.1",
    author="Yilang Zhang, Bingcong Li, Niao He, Georgios B. Giannakis",
    description="Adaptive Neural Connection Reassignment (ANCRe)",
    long_description_content_type="text/markdown",
    url="https://github.com/zhangyilang/ANCRe",
    license="Apache-2.0",
    packages=["ancre"],
    install_requires=["torch"],
    extras_require={
        # pinned environment for reproducing the paper (llm/, dm/)
        "experiments": [
            "torch==2.8.0",
            "torchvision==0.23.0",
            "transformers",
            "tokenizers",
            "datasets==3.6.0",
            "evaluate",
            "accelerate",
            "bitsandbytes",
            "galore-torch==1.0",
            "wandb",
            "loguru",
            "timm",
            "diffusers",
            # for evaluation of diffusion models
            "tensorflow>=2.0",
            "scipy",
            "requests",
            "tqdm",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)