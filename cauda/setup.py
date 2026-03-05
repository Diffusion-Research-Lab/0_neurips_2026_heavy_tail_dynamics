from pathlib import Path
from setuptools import find_packages, setup


def read_requirements(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


setup(
    name="cauda",
    version="0.1.0",
    description="cauda package",
    python_requires=">=3.10",
    packages=find_packages(include=["cauda", "cauda.*"]),
    install_requires=read_requirements("requirements.txt"),
)
