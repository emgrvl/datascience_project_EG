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
        "pandas>=2.3.3",
        "numpy>=2.3.5",
        "scikit-learn>=1.7.2",
        "matplotlib>=3.10.7",
        "seaborn>=0.13.2",
        "rdkit>=2025.9.1",
        "imbalanced-learn>=0.14.0",
        "plotly>=6.4.0",
        "streamlit>=1.51.0",
        "tqdm>=4.67.1",
        "requests>=2.32.5",
        "beautifulsoup4>=4.14.2",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Machine Learning",
        "Intended Audience :: Science/Research",
    ],
    entry_points={
        "console_scripts": [
            "deepscent-train=src.train:main",
            "deepscent-eval=src.evaluate:main",
        ],
    },
)

