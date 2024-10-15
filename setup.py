from setuptools import setup

with open('README.md') as file_:
    long_description = file_.read()

setup(
    name = 'ghnn',
    version = 1.0,
    author = 'Philipp Horn',
    author_email = 'p.horn@tue.nl',
    description = 'Generalized Hamiltonian Neural Networks',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/AELITTEN/GHNN',
    license = 'MIT License',
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    packages = ['ghnn'],
    python_requires = '>=3.6',
    install_requires = [
        'tables',
        'numpy <2.0',
        'pandas',
        'scipy',
        'torch >=2.0',
        'matplotlib'
    ]
)
