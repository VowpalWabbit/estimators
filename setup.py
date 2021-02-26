import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="vw-estimators",
    version="0.0.1",
    description="Python package of estimators to perform off-policy evaluation ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VowpalWabbit/estimators.git",
    license="BSD 3-Clause License",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    packages=["estimators"],
    install_requires= ['scipy>=0.9'],
    tests_require=['pytest'],
    python_requires=">=3.6",
)