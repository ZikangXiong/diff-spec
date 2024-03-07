from setuptools import find_packages, setup

with open("requirements.txt") as f:
    REQUIRED_PACKAGES = f.read().splitlines()

setup(
    name="diff-spec",
    version="0.2.0",
    description="Differentiable Symbolic Specification",
    author="Zikang Xiong",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=REQUIRED_PACKAGES,
)
