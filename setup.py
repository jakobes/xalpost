# System imports
from setuptools import setup

setup(
    name = "postprocessing",
    author = "Jakob E. Schreiner",
    author_email = "jakob@xal.no",
    packages = ["xalpost", "postspec"],
    package_dir = {"": "src"} 
)
