from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="bedrock_llm",
    version="0.1.3.5",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    author="Tran Quy An",
    author_email="an.tq@techxcorp.com",
    description="A Python LLM framework for interacting with AWS Bedrock services. Build on top of boto3 library. This library serves as an fast prototyping, building POC, production ready.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Phicks-debug/bedrock_llm",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires='>=3.9',
    keywords="aws bedrock llm machine-learning ai",
)