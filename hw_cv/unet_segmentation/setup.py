from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="unet_segmentation",
    version="1.0",
    description="A Python machine learning module for image segmentation",
    author="Alex Nemkovich",
    author_email="alexey3nemckovich@gmail.com",
    packages=find_packages(),
    install_requires=requirements,
)
