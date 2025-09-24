import setuptools

with open("README.md", "r",encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPOSITORY_NAME = "Classifier"
AUTHOR_USER_NAME = "mehraj-alom"
AUTHOR_EMAIL = "mehraj.official000@gmail.com"
SRC_REPO = "src"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPOSITORY_NAME}",
    #project_urls={
    
    package_dir={"": SRC_REPO},
    packages=setuptools.find_packages(where=SRC_REPO),
)