# python setup.py build_ext --inplace

import pathlib
from setuptools import Extension, dist, find_packages, setup

dist.Distribution().fetch_build_eggs(["numpy>=1.10"])

# these two imports must be below the line above; which ensures they're available
# for use during installation
import numpy  # isort:skip

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

exec(open("plaster/version.py").read())

setup(
    name="erisyon.plaster",
    version=__version__,
    description="Erisyon's Fluoro-Sequencing Platform",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/erisyon/plaster",
    author="Erisyon",
    author_email="plaster+pypi@erisyon.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["plaster"],
    include_package_data=True,
    install_requires=[
        "arrow",
        "bokeh",
        "ipython",
        "jupyter",
        "munch",
        "nbstripout",
        "nptyping",
        "numpy",
        "opencv-python",
        "pandas==1.0.5",
        "plumbum",
        # see the comment in plaster/plaster/run/sigproc_v2/synth.py for why this is commented out
        # "psf",
        "pudb",
        "pyyaml",
        "requests",
        "retrying",
        "scikit-image",
        "scikit-learn",
        "twine",
        "wheel",
        "zbs.zest",
    ],
    python_requires=">=3.6",
)
