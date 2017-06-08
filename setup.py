import logging as log
import os
import platform
import sys
from distutils.command.build import build as _build
from distutils.errors import DistutilsSetupError
from distutils.spawn import find_executable
from distutils.sysconfig import get_python_lib
from multiprocessing import cpu_count
from shutil import rmtree
from subprocess import Popen

from setuptools import setup


def run_process(cmds):
    p = Popen(cmds)
    p.wait()
    return p.returncode


log.basicConfig(stream=sys.stdout, level=log.INFO)

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
    log.info("Removing " + d)
    rmtree(d)


class build(_build):
    def __init__(self, *args, **kwargs):
        _build.__init__(self, *args, **kwargs)
        self.script_dir = None
        self.build_dir = None
        self.cmake_path = None
        self.make_path = None
        self.hg_path = None
        self.cxx_path = None
        self.cc_path = None
        self.install_prefix = None
        self.py_executable = None
        self.py_version = None

    def run(self):
        py_executable = sys.executable
        py_version = "%s.%s" % (sys.version_info[0], sys.version_info[1])
        build_name = "py%s-%s" % (py_version, platform.architecture()[0])
        build_dir = os.path.join(script_dir, "build", build_name)

        self.script_dir = script_dir
        self.build_dir = build_dir
        self.cmake_path = os.environ.get("CMAKE") or find_executable("cmake")
        if not self.cmake_path:
            raise DistutilsSetupError("`cmake` not found, and `CMAKE` is not set.")
        self.make_path = os.environ.get("MAKE") or find_executable("make")
        if not self.make_path:
            raise DistutilsSetupError("`make` not found, and `MAKE` is not set.")
        self.hg_path = find_executable("hg")
        if not self.hg_path:
            raise DistutilsSetupError("`hg` not found.")
        self.cc_path = os.environ.get("CC") or find_executable("gcc")
        if not self.cc_path:
            raise DistutilsSetupError("`gcc` not found, and `CC` is not set.")
        self.cxx_path = os.environ.get("CXX") or find_executable("g++")
        if not self.cxx_path:
            raise DistutilsSetupError("`g++` not found, and `CXX` is not set.")
        self.install_prefix = os.path.join(get_python_lib(), os.pardir, os.pardir, os.pardir)
        self.py_executable = py_executable
        self.py_version = py_version

        log.info("=" * 30)
        log.info("CMake path: " + self.cmake_path)
        log.info("Make path: " + self.make_path)
        log.info("Mercurial path: " + self.hg_path)
        log.info("C compiler path: " + self.cc_path)
        log.info("CXX compiler path: " + self.cxx_path)
        log.info("-" * 3)
        log.info("Script directory: " + self.script_dir)
        log.info("Build directory: " + self.build_dir)
        log.info("Library installation directory: " + self.install_prefix)
        log.info("Python executable: " + self.py_executable)
        log.info("=" * 30)

        # Prepare folders
        if not os.path.exists(self.build_dir):
            log.info("Creating build directory " + self.build_dir)
            os.makedirs(self.build_dir)

        os.chdir(self.build_dir)

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
            "-DCMAKE_INSTALL_PREFIX=" + self.install_prefix,
            "-DEIGEN3_INCLUDE_DIR=" + os.path.join(self.build_dir, "eigen"),
            "-DPYTHON=" + self.py_executable,
            ]
        boost_prefix = os.environ.get("BOOST")
        if boost_prefix:
            cmake_cmd += [
                "-DBOOST_ROOT:PATHNAME=" + boost_prefix,
                "-DBoost_NO_BOOST_CMAKE=TRUE",
                "-DBoost_NO_SYSTEM_PATHS=TRUE",
                "-DBoost_LIBRARY_DIRS:FILEPATH=" + boost_prefix + "/lib",
            ]
        log.info("Configuring...")
        if run_process(cmake_cmd) != 0:
            raise DistutilsSetupError(" ".join(cmake_cmd))

        make_cmd = [self.make_path, "-j", str(cpu_count())]
        log.info("Compiling...")
        if run_process(make_cmd) != 0:
            raise DistutilsSetupError(" ".join(make_cmd))

        make_cmd = [self.make_path, "install"]
        log.info("Installing...")
        if run_process(make_cmd) != 0:
            raise DistutilsSetupError(" ".join(make_cmd))

        setup_cmd = [sys.executable, "setup.py", "install"]
        log.info("Installing Python modules...")
        os.chdir("python")
        if run_process(setup_cmd) != 0:
            raise DistutilsSetupError(" ".join(setup_cmd))

        os.chdir(self.script_dir)


try:
    import pypandoc
    long_description = pypandoc.convert("README.md", "rst")
except (IOError, ImportError):
    long_description = ""

setup(
    name="dyNET",
    #version="0.0.0",
    install_requires=["cython", "numpy"],
    description="The Dynamic Neural Network Toolkit",
    long_description=long_description,
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
    cmdclass={"build": build},
)
