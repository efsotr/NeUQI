from setuptools import setup, find_packages

setup(
    name="optimized_module",
    version="1.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"}
)