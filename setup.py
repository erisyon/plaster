# python setup.py build_ext --inplace

import pathlib
from setuptools import Extension, dist, find_packages, setup

dist.Distribution().fetch_build_eggs(["Cython>=0.15.1", "numpy>=1.10"])

# these two imports must be below the line above; which ensures they're available
# for use during installation
from Cython.Build import cythonize  # isort:skip
import numpy  # isort:skip

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

exec(open("plaster/version.py").read())

extensions = [
    Extension(
        name="plaster.run.sim_v2.fast.sim_v2_fast",
        sources=[
            "./plaster/run/sim_v2/fast/sim_v2_fast.pyx",
            "./plaster/run/sim_v2/fast/c_sim_v2_fast.c",
            "./plaster/tools/c_common/c_common.c",
        ],
        include_dirs=[
            "./plaster/run/sim_v2/fast",
            "./plaster/tools/c_common",
            numpy.get_include(),
        ],
        extra_compile_args=[
            "-Wno-unused-but-set-variable",
            "-Wno-unused-label",
            "-Wno-cpp",
            "-pthread",
            "-DDEBUG",
            # "-DNDEBUG",
        ],
    ),
    Extension(
        name="plaster.run.survey_v2.fast.survey_v2_fast",
        sources=[
            "./plaster/run/survey_v2/fast/survey_v2_fast.pyx",
            "./plaster/run/survey_v2/fast/c_survey_v2_fast.c",
            "./plaster/tools/c_common/c_common.c",
        ],
        include_dirs=[
            "./plaster/run/survey_v2/fast",
            "./plaster/tools/c_common",
            "/flann/src/cpp/flann/",
            numpy.get_include(),
        ],
        libraries=["flann"],
        library_dirs=["/flann/lib"],
        extra_compile_args=[
            "-DNPY_NO_DEPRECATED_API",
            "-DDEBUG",
            # "-DNDEBUG",
        ],
    ),
]


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
        "cython",
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
    ext_modules=cythonize(
        extensions, language_level="3", include_path=["./plaster/tools/c_common",]
    ),
)
