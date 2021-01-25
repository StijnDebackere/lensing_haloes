import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lensing_haloes", # Replace with your own username
    version="0.0.1",
    author="Stijn Debackere",
    author_email="debackere@strw.leidenuniv.nl",
    description="A package to generate cluster samples and weak lensing fits.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StijnDebackere/lensing_haloes/",
    packages=setuptools.find_packages(),
    install_requires=[
        "asdf>=2.7.1",
        "astropy",
        "emcee",
        "george @ git+https://github.com/StijnDebackere/george",
        "mpmath",
        "numpy",
        "pyccl",
        "scipy",
        "tqdm",
        "tremulator",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
