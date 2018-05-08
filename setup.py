# System imports
from setuptools import setup

setup(
    name = "postprocessing",
    author = "Jakob E. Schreiner",
    author_email = "jakob@xal.no",
    packages = ["post", "postspec", "postfields", "postutils"],
    package_dir = {"": "src"} 
)
