from setuptools import setup

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='portfolio_backtester',
    version='0.1.0',
    packages=['tests', 'portfolio_backtester'],
    keywords=['portfolio', 'backtest'],
    url='',
    license='MIT',
    author='zhudechuan',
    author_email='dz2414@columbia.edu',
    description='universal portfolio backtester',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    package_data={"portfolio_backtester": ["data/*.txt", "data/*.csv"]},
    install_requires=["numpy", "pandas", "scipy", "scikit-learn"]
)
