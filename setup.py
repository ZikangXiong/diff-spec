from setuptools import find_packages, setup

with open("requirements.txt") as f:
    REQUIRED_PACKAGES = f.read().splitlines()

setup(
    name="differntiable-symbolic-specification",
    version="0.1.0",
    description="Differentiable Symbolic Specification",
    author="Zikang Xiong",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    requires=REQUIRED_PACKAGES,
)
