from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow-gpu==2.12.0',
    'h5py==2.10.0',
    'numpy>=1.11.0',
    'nibabel>=2.2.1',
    'statsmodels>=0.8.0',
    'tqdm>=4.15.0',
    'scipy>=0.18.1',
    'scikit_learn>=0.19.0']

setup(
    name='flexconn_train',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='FLEXCONN code'
)