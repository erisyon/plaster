# python setup.py build_ext --inplace

import os
import pathlib
from setuptools import Extension, dist, find_packages, setup
from setuptools.command.build_ext import build_ext as build_ext_orig

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
        ],
        include_dirs=["./plaster/run/sim_v2/fast", numpy.get_include(),],
        extra_compile_args=[
            "-Wno-unused-but-set-variable",
            "-Wno-unused-label",
            "-Wno-cpp",
            "-pthread",
            # "-DNDEBUG",
        ],
    ),
    Extension(
        name="plaster.run.nn_v2.fast.nn_v2_fast",
        sources=[
            "./plaster/run/nn_v2/fast/nn_v2_fast.pyx",
            "./plaster/run/nn_v2/fast/c_nn_v2_fast.c",
        ],
        include_dirs=[
            "./plaster/run/nn_v2/fast",
            "/flann/src/cpp/flann/",
            numpy.get_include(),
        ],
        libraries=["flann"],
        library_dirs=["/flann/lib"],
        extra_compile_args=[
            "-DNPY_NO_DEPRECATED_API",
            # "-DNDEBUG",
        ],
    ),
]


# CMake logic copied from https://stackoverflow.com/a/48015772
class CMakeExtension(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config
        ]

        # example of build args
        build_args = [
            '--config', config,
            '--', '-j4'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        # Troubleshooting: if fail on line above then delete all possible
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))


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
        "pandas",
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
    #ext_modules=[CMakeExtension('./plaster/vendor/flann')] + cythonize(extensions, language_level="3"),
    ext_modules=[CMakeExtension('flann')] #+ cythonize(extensions, language_level="3"),
)
