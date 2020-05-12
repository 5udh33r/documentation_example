import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="e-calculator", # Replace with your own username
    version="0.0.2",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bfollinprm/documentation_example",
    packages=setuptools.find_packages(),
    install_requires=["matplotlib", "numpy"]
)
