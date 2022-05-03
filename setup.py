from setuptools import setup, find_packages

setup(
    name='kore2022',
    version='1.0',
    packages=find_packages(
        where='kformer',
    ),
    install_requires=[
        "pandas",  
        "scikit-learn",
        "black",
        "pympler"
    ],
)