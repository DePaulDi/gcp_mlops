# setup.py
from setuptools import find_packages
from setuptools import setup

setup(
    name='trainer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'gcsfs'  
        'python-json-logger'  
    ],
    description='Crop Yield LSTM Training Package'
)