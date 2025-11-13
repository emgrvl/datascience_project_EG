#Set up
from setuptools import setup, find_packages

setup(
    name="DeepScent",
    version="1.0.0",
    author="Emeline Gravaillac",
    author_email="emeline.gravaillac@unil.ch",
    description="DeepScent – Predicting fragrance families from molecular architectures",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/emgrvl/datascience_project_EG",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.5.0",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.0",
        "rdkit-pypi>=2023.9.4",
        "imbalanced-learn>=0.12.0",
        "plotly>=5.22.0",
        "streamlit>=1.38.0",
        "tqdm>=4.66.0",
        "requests>=2.32.0",
        "beautifulsoup4>=4.12.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    entry_points={
        "console_scripts": [
            "deepscent-train=src.model_training:main",
            "deepscent-eval=src.evaluation:main",
        ],
    },
)

