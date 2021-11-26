from setuptools import find_packages, setup

setup(
    name="personalized_doc_summ",
    version="0.1",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    entry_points={"console_scripts": ["personalized_doc_summ = src.personalized_doc_summarization.main:main"]},
)