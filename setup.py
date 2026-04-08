from setuptools import find_packages
from setuptools import setup

setup(
    name='trainer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'gcsfs',      # Required for pd.read_csv('gs://...')
        'numpy',
        'scikit-learn'
    ],
    include_package_data=True,
    description='LSTM Training Package'
)