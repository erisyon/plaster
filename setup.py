# python setup.py build_ext --inplace

import sys
import os
import subprocess
import pathlib
from pathlib import Path
from setuptools import Extension, dist, find_packages, setup
from setuptools.command.build_ext import build_ext

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


# CMake logic copied from https://gist.github.com/ossareh/38ea13049d23f6742f78046170f4c033#file-setup-py-L86
class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])

# TODO: Not working yet, there's something wrong with the paths where the so goes.
class CMakeBuild(build_ext):
    def run(self):
        os.chdir("/erisyon/plaster/plaster/vendor/flann")
        
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        build_directory = os.path.abspath(self.build_temp)

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + build_directory,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        # Assuming Makefiles
        build_args += ['--', '-j2']

        self.build_args = build_args

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print('-' * 10, 'Running CMake prepare', '-' * 40)
        subprocess.check_call(['cmake', cmake_list_dir] + cmake_args,
                              cwd=self.build_temp, env=env)

        print('-' * 10, 'Building extensions', '-' * 40)
        cmake_cmd = ['cmake', '--build', '.'] + self.build_args
        subprocess.check_call(cmake_cmd,
                              cwd=self.build_temp)

        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext):
        build_temp = Path(self.build_temp).resolve()
        dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        source_path = build_temp / self.get_ext_filename(ext.name)
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(source_path, dest_path)


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
    cmdclass=dict(build_ext=CMakeBuild),
    python_requires=">=3.6",
    ext_modules=[CMakeExtension('plaster.vendor.flann')] #+ cythonize(extensions, language_level="3"),
)
