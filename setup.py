from setuptools import setup, find_packages


setup(
    name = "xalpost",
    author = "Jakob E. Schreiner",
    author_email = "jakob@xal.no",
    packages = find_packages("src"),
    package_dir = {"": "src"},
    install_requires=[
        "pandas",
        "pyyaml",
    ]
)
