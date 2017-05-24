import logging as log
import os
import platform
import sys
from distutils.command.build import build as _build
from distutils.command.build_ext import build_ext as _build_ext
from distutils.errors import DistutilsSetupError
from distutils.spawn import find_executable
from distutils.sysconfig import get_python_lib
from shutil import rmtree
from subprocess import Popen

from setuptools import setup, Extension
from setuptools.command.bdist_egg import bdist_egg as _bdist_egg
from setuptools.command.develop import develop as _develop


def run_process(cmds):
    p = Popen(cmds)
    p.wait()
    return p.returncode

__version__ = "1.0.0"


# Change the cwd to our source dir
try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = os.path.abspath(this_file)
if os.path.dirname(this_file):
    os.chdir(os.path.dirname(this_file))
script_dir = os.getcwd()

# Clean up temp and package folders
d = os.path.join(script_dir, "build")
if os.path.isdir(d):
    print("Removing %s" % d)
    rmtree(d)


class dynet_develop(_develop):
    def __init__(self, *args, **kwargs):
        _develop.__init__(self, *args, **kwargs)

    def run(self):
        self.run_command("build")
        _develop.run(self)


class dynet_bdist_egg(_bdist_egg):
    def __init__(self, *args, **kwargs):
        _bdist_egg.__init__(self, *args, **kwargs)

    def run(self):
        self.run_command("build")
        _bdist_egg.run(self)


class dynet_build_ext(_build_ext):
    def __init__(self, *args, **kwargs):
        _build_ext.__init__(self, *args, **kwargs)

    def run(self):
        pass


class dynet_build(_build):
    def __init__(self, *args, **kwargs):
        _build.__init__(self, *args, **kwargs)
        self.script_dir = None
        self.build_dir = None
        self.cmake_path = None
        self.make_path = None
        self.hg_path = None
        self.cxx_path = None
        self.cc_path = None
        self.site_packages_dir = None
        self.py_executable = None
        self.py_version = None

    def initialize_options(self):
        _build.initialize_options(self)

    def run(self):
        py_executable = sys.executable
        py_version = "%s.%s" % (sys.version_info[0], sys.version_info[1])
        build_name = "py%s-%s" % (py_version, platform.architecture()[0])
        build_dir = os.path.join(script_dir, "build", "%s" % build_name)

        self.script_dir = script_dir
        self.build_dir = build_dir
        self.cmake_path = find_executable("cmake")
        self.make_path = find_executable("make")
        self.hg_path = find_executable("hg")
        self.cxx_path = find_executable("g++-4.9") or find_executable("g++-4.8")
        self.cc_path = find_executable("gcc-4.9") or find_executable("gcc-4.8")
        self.site_packages_dir = get_python_lib(1, 0, prefix=build_dir)
        self.py_executable = py_executable
        self.py_version = py_version

        # Prepare folders
        if not os.path.exists(self.build_dir):
            log.info("Creating build directory %s..." % self.build_dir)
            os.makedirs(self.build_dir)

        log.basicConfig(filename=os.path.join(build_dir, "setup.log"), level=log.INFO)
        os.chdir(self.build_dir)

        log.info("=" * 30)
        log.info("Package version: %s" % __version__)
        log.info("-" * 3)
        log.info("CMake path: %s" % self.cmake_path)
        log.info("Make path: %s" % self.make_path)
        log.info("Mercurial path: %s" % self.hg_path)
        log.info("-" * 3)
        log.info("Script directory: %s" % self.script_dir)
        log.info("Build directory: %s" % self.build_dir)
        log.info("Python site-packages install directory: %s" % self.site_packages_dir)
        log.info("Python executable: %s" % self.py_executable)
        log.info("=" * 30)

        hg_cmd = [self.hg_path, "clone", "https://bitbucket.org/eigen/eigen"]
        log.info("Cloning Eigen...")
        if run_process(hg_cmd) != 0:
            raise DistutilsSetupError(" ".join(hg_cmd))

        os.environ["CXX"] = self.cxx_path
        os.environ["CC"] = self.cc_path

        # Build module
        cmake_cmd = [
            self.cmake_path,
            script_dir,
            "-DEIGEN3_INCLUDE_DIR=eigen",
            "-DPYTHON=%s" % self.py_executable,
            ]
        boost_prefix = os.environ.get("BOOST")
        if boost_prefix:
            cmake_cmd += [
                "-DBOOST_ROOT:PATHNAME=" + boost_prefix,
                "-DBoost_NO_BOOST_CMAKE=TRUE",
                "-DBoost_NO_SYSTEM_PATHS=TRUE,",
                "-DBoost_LIBRARY_DIRS:FILEPATH=" + boost_prefix + "/lib",
            ]
        log.info("Configuring...")
        if run_process(cmake_cmd) != 0:
            raise DistutilsSetupError(" ".join(cmake_cmd))

        make_cmd = [self.make_path]
        log.info("Compiling...")
        if run_process(make_cmd) != 0:
            raise DistutilsSetupError(" ".join(make_cmd))

        setup_cmd = [self.py_executable, "setup.py", "install"]
        log.info("Installing...")
        os.chdir("python")
        if run_process(setup_cmd) != 0:
            raise DistutilsSetupError(" ".join(setup_cmd))

        os.chdir(self.script_dir)


try:
    with open(os.path.join(script_dir, "README.md")) as f:
        README = f.read()
except IOError:
    README = ""

setup(
    name="dynet",
    version=__version__,
    install_requires=["cython", "numpy"],
    description="The Dynamic Neural Network Toolkit",
    long_description=README,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Environment :: MacOS X",
        "Environment :: Win32 (MS Windows)",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    author="Graham Neubig",
    author_email="dynet-users@googlegroups.com",
    url="https://github.com/clab/dynet",
    download_url="https://github.com/clab/dynet/releases",
    license="Apache 2.0",
    include_package_data=True,
    zip_safe=False,
    cmdclass={
        "build": dynet_build,
        "build_ext": dynet_build_ext,
        "bdist_egg": dynet_bdist_egg,
        "develop": dynet_develop,
    },
    ext_modules=[Extension("_dynet", [])],
    ext_package="dyNET",
)
