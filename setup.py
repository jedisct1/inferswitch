"""
Setup script for InferSwitch.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="inferswitch",
    version="0.1.0",
    author="InferSwitch Team",
    description="An Anthropic API proxy with logging and chat template support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "fastapi>=0.115.12",
        "httpx>=0.28.1", 
        "pydantic>=2.11.5",
        "uvicorn>=0.34.3",
    ],
    entry_points={
        "console_scripts": [
            "inferswitch=inferswitch.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)