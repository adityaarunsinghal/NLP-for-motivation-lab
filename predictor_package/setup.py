import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="predictor_motivation_lab", 
    version="0.1.0",
    author="Aditya Singhal",
    author_email="adis@nyu.edu",
    description="Predictor for NYU Motivation Lab",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adityaarunsinghal/NLP-for-motivation-lab",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)