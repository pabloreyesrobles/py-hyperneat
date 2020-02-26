from setuptools import setup
from setuptools import find_packages

setup(
    name='py-hyperneat', # Replace with your own username
    version='0.0.1',
    author='Pablo Reyes Robles',
    author_email='pabloreyes500@gmail.com',
    description='An HyperNEAT implementation for Python.',
    url='https://gitlab.com/pablo_rr/py-hyperneat',
    packages=['neat', 'hyperneat'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)