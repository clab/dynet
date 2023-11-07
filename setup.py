import distutils.sysconfig
import logging as log
import platform
import zipfile
import sys
from distutils.command.build import build as _build
from distutils.command.build_py import build_py as _build_py
from distutils.command.install_data import install_data as _install_data
from distutils.errors import DistutilsSetupError
from distutils.spawn import find_executable
from distutils.sysconfig import get_python_lib
from multiprocessing import cpu_count
from subprocess import Popen

import os
import re
from Cython.Distutils import build_ext as _build_ext
from setuptools import setup
from setuptools.extension import Extension
from shutil import rmtree, copytree, copy

# urlretrieve has a different location in Python 2 and Python 3
import urllib
if hasattr(urllib, "urlretrieve"):
    urlretrieve = urllib.urlretrieve
else:
    import urllib.request
    urlretrieve = urllib.request.urlretrieve


def run_process(cmds):
    p = Popen(cmds)
    p.wait()
    return p.returncode


def append_cmake_list(l, var):
    if var:
        l.extend(var.split(";"))


def append_cmake_lib_list(l, var):
    if var:
        l.extend(map(strip_lib, var.split(";")))


# Strip library prefixes and suffixes to prevent linker confusion
def strip_lib(filename):
    filename = re.sub(r"^(?:lib)?(.*)\.(?:so|a|dylib)$", r"\1", filename)
    filename = re.sub(r"^(.*)\.lib$", r"\1", filename)
    return filename

def get_env(build_dir):

  # Get environmental variables first
  ENV = dict(os.environ)

  # Get values listed in the CMakeCache.txt file (if existant)
  try:
      var_regex = r"^([^:]+):([^=]+)=(.*)$"
      cache_path = os.path.join(build_dir, "CMakeCache.txt")
      with open(cache_path, "r") as cache_file:
          for line in cache_file:
              line = line.strip()
              m = re.match(var_regex, line)
              if m:
                  ENV[m.group(1)] = m.group(3)
  except:
      pass

  # Get values passed on the command line
  i = 0
  for i, arg in enumerate(sys.argv[1:]):
      try:
          key, value = arg.split("=", 1)
      except ValueError:
          break
      ENV[key] = value 
  del sys.argv[1:i+1]

  return ENV

log.basicConfig(stream=sys.stdout, level=log.INFO)

# Find the current directory
try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
ORIG_DIR = os.getcwd()
SCRIPT_DIR = os.path.dirname(os.path.abspath(this_file))
if ORIG_DIR.rstrip('/').endswith('python'):
    BUILD_DIR = ORIG_DIR.rstrip('/').rstrip('python')
    PYTHON_DIR = ORIG_DIR
else:
    BUILD_DIR = ORIG_DIR
    PYTHON_DIR = ORIG_DIR + '/python'
ENV = get_env(BUILD_DIR)

# Find the paths
BUILT_EXTENSIONS = False
CMAKE_PATH = ENV.get("CMAKE", find_executable("cmake"))
MAKE_PATH = ENV.get("MAKE", find_executable("make"))
MAKE_FLAGS = ENV.get("MAKE_FLAGS", "-j %d" % cpu_count()).split()
CC_PATH = ENV.get("CC", find_executable("gcc"))
CXX_PATH = ENV.get("CXX", find_executable("g++"))
INSTALL_PREFIX = os.path.join(get_python_lib(), os.pardir, os.pardir, os.pardir)
PYTHON = sys.executable

# Try to find Eigen
EIGEN3_INCLUDE_DIR = ENV.get("EIGEN3_INCLUDE_DIR")  # directory where eigen is saved
# The cmake directory and Python directory are different in manual install, so
# will break if relative path is specified. Try moving up if path is specified
# but not found
if (EIGEN3_INCLUDE_DIR is not None and
    not os.path.isdir(EIGEN3_INCLUDE_DIR) and
    os.path.isdir(os.path.join(os.pardir, EIGEN3_INCLUDE_DIR))):
    EIGEN3_INCLUDE_DIR = os.path.join(os.pardir, EIGEN3_INCLUDE_DIR)

EIGEN3_DOWNLOAD_URL = ENV.get("EIGEN3_DOWNLOAD_URL", "https://github.com/clab/dynet/releases/download/2.1/eigen-b2e267dc99d4.zip") 
    
# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
cfg_vars = distutils.sysconfig.get_config_vars()
CFLAGS = cfg_vars.get("CFLAGS")
if CFLAGS is not None:
    cfg_vars["CFLAGS"] = CFLAGS.replace("-Wstrict-prototypes", "")

# For Cython extensions
LIBRARIES = ["dynet"]
LIBRARY_DIRS = ["."]
COMPILER_ARGS = []
EXTRA_LINK_ARGS = []
RUNTIME_LIB_DIRS = []
INCLUDE_DIRS = []
DATA_FILES=[]

# Add all environment variables from CMake for Cython extensions
append_cmake_lib_list(LIBRARIES, ENV.get("CUDA_CUBLAS_FILES"))
append_cmake_list(LIBRARY_DIRS, ENV.get("CUDA_CUBLAS_DIRS"))
CMAKE_INSTALL_PREFIX = ENV.get("CMAKE_INSTALL_PREFIX", INSTALL_PREFIX)
LIBS_INSTALL_DIR = CMAKE_INSTALL_PREFIX + "/lib/"
PROJECT_SOURCE_DIR = ENV.get("PROJECT_SOURCE_DIR", SCRIPT_DIR)  # location of the main dynet directory
PROJECT_BINARY_DIR = ENV.get("PROJECT_BINARY_DIR", BUILD_DIR)  # path where dynet is built
DYNET_LIB_DIR = PROJECT_BINARY_DIR + "/dynet/"

if ENV.get("MSVC") == "1":
    COMPILER_ARGS[:] = ["-DNOMINMAX", "/EHsc"]
    DYNET_LIB_DIR += "/Release/"
    # For MSVC, we compile dynet as a static lib, so we need to also link in the
    # other libraries it depends on:
    append_cmake_lib_list(LIBRARIES, ENV.get("LIBS"))
    append_cmake_list(LIBRARY_DIRS, ENV.get("MKL_LINK_DIRS"))  # Add the MKL dirs, if MKL is being used
    append_cmake_lib_list(LIBRARIES, ENV.get("CUDA_RT_FILES"))
    append_cmake_list(LIBRARY_DIRS, ENV.get("CUDA_RT_DIRS"))
    DATA_FILES += [DYNET_LIB_DIR + lib + ".lib" for lib in LIBRARIES]
else:
    COMPILER_ARGS[:] = ["-std=c++11", "-Wno-unused-function"]
    RUNTIME_LIB_DIRS.extend([DYNET_LIB_DIR, LIBS_INSTALL_DIR])
    # in some OSX systems, the following extra flags are needed:
    if platform.system() == "Darwin":
        COMPILER_ARGS.extend(["-stdlib=libc++", "-mmacosx-version-min=10.7"])
        EXTRA_LINK_ARGS.append("-Wl,-rpath," + LIBS_INSTALL_DIR)
        if "--skip-build" not in sys.argv:  # Include libdynet.dylib unless doing manual install
            DATA_FILES += [os.path.join(LIBS_INSTALL_DIR, "lib%s.dylib" % lib) for lib in LIBRARIES]
    else:
        EXTRA_LINK_ARGS.append("-Wl,-rpath=%r" % LIBS_INSTALL_DIR + ",--no-as-needed")

LIBRARY_DIRS.insert(0, DYNET_LIB_DIR)

INCLUDE_DIRS[:] = filter(None, [PROJECT_SOURCE_DIR, EIGEN3_INCLUDE_DIR])

TARGET = [Extension(
    "_dynet",  # name of extension
    [PYTHON_DIR + "/_dynet.pyx"],  # filename of our Pyrex/Cython source
    language="c++",  # this causes Pyrex/Cython to create C++ source
    include_dirs=INCLUDE_DIRS,
    libraries=LIBRARIES,
    library_dirs=LIBRARY_DIRS,
    extra_link_args=EXTRA_LINK_ARGS,
    extra_compile_args=COMPILER_ARGS,
    runtime_library_dirs=RUNTIME_LIB_DIRS,
)]

class build(_build):
    user_options = [
        ("build-dir=", None, "New or existing DyNet build directory."),
        ("skip-build", None, "Assume DyNet C++ library is already built."),
    ]
        
    def __init__(self, *args, **kwargs):
        self.build_dir = None
        self.skip_build = False
        _build.__init__(self, *args, **kwargs)

    def initialize_options(self):
        py_version = "%s.%s" % (sys.version_info[0], sys.version_info[1])
        unicode_suffix = "u" if sys.version_info[0] == 2 and sys.maxunicode > 65536 else ""
        build_name = "py%s%s-%s" % (py_version, unicode_suffix, platform.architecture()[0])
        self.build_dir = os.path.join(SCRIPT_DIR, "build", build_name)
        _build.initialize_options(self)

    def run(self):
        global BUILD_DIR, BUILT_EXTENSIONS, EIGEN3_INCLUDE_DIR
        BUILD_DIR = os.path.abspath(self.build_dir)
        if EIGEN3_INCLUDE_DIR is None:
            EIGEN3_INCLUDE_DIR = os.path.join(BUILD_DIR, "eigen")
        EIGEN3_INCLUDE_DIR = os.path.abspath(EIGEN3_INCLUDE_DIR)    
        log.info("CMAKE_PATH=%r" % CMAKE_PATH)
        log.info("MAKE_PATH=%r" % MAKE_PATH)
        log.info("MAKE_FLAGS=%r" % " ".join(MAKE_FLAGS))
        log.info("EIGEN3_INCLUDE_DIR=%r" % EIGEN3_INCLUDE_DIR)
        log.info("EIGEN3_DOWNLOAD_URL=%r" % EIGEN3_DOWNLOAD_URL)
        log.info("CC_PATH=%r" % CC_PATH)
        log.info("CXX_PATH=%r" % CXX_PATH)
        log.info("SCRIPT_DIR=%r" % SCRIPT_DIR)
        log.info("BUILD_DIR=%r" % BUILD_DIR)
        log.info("INSTALL_PREFIX=%r" % INSTALL_PREFIX)
        log.info("PYTHON=%r" % PYTHON)
        if CMAKE_PATH is not None:
            run_process([CMAKE_PATH, "--version"])
        if CXX_PATH is not None:
            run_process([CXX_PATH, "--version"])

        # This will generally be called by the pip install
        if not self.skip_build:
            if CMAKE_PATH is None:
                raise DistutilsSetupError("`cmake` not found, and `CMAKE` is not set.")
            if MAKE_PATH is None:
                raise DistutilsSetupError("`make` not found, and `MAKE` is not set.")
            if CC_PATH is None:
                raise DistutilsSetupError("`gcc` not found, and `CC` is not set.")
            if CXX_PATH is None:
                raise DistutilsSetupError("`g++` not found, and `CXX` is not set.")

            # Prepare folders
            if not os.path.isdir(BUILD_DIR):
                log.info("Creating build directory " + BUILD_DIR)
                os.makedirs(BUILD_DIR)

            os.chdir(BUILD_DIR)
            if os.path.isdir(EIGEN3_INCLUDE_DIR):
                log.info("Found eigen in " + EIGEN3_INCLUDE_DIR)
            else:
                try:
                    # Can use BZ2 or zip, right now using zip
                    # log.info("Fetching Eigen...")
                    # urlretrieve(EIGEN3_DOWNLOAD_URL, "eigen.tar.bz2")
                    # log.info("Unpacking Eigen...")
                    # tfile = tarfile.open("eigen.tar.bz2", 'r')
                    # tfile.extractall('eigen')
                    log.info("Fetching Eigen...")
                    urlretrieve(EIGEN3_DOWNLOAD_URL, "eigen.zip")
                except Exception as e:
                    raise DistutilsSetupError("Could not download Eigen from %r: %s" % (EIGEN3_DOWNLOAD_URL, e))
                try:
                    log.info("Unpacking Eigen...")
                    os.mkdir(EIGEN3_INCLUDE_DIR)
                    with zipfile.ZipFile("eigen.zip") as zfile:
                        zfile.extractall(EIGEN3_INCLUDE_DIR)
                except Exception as e:
                    raise DistutilsSetupError("Could not extract Eigen to %r: %s" % (EIGEN3_INCLUDE_DIR, e))

            os.environ["CXX"] = CXX_PATH
            os.environ["CC"] = CC_PATH

            # Build module
            cmake_cmd = [
                CMAKE_PATH,
                SCRIPT_DIR,
                "-DCMAKE_INSTALL_PREFIX=%r" % INSTALL_PREFIX,
                "-DEIGEN3_INCLUDE_DIR=%r" % EIGEN3_INCLUDE_DIR,
                "-DPYTHON=%r" % PYTHON,
            ]
            for env_var in ("BACKEND", "CUDNN_ROOT", "CUDA_TOOLKIT_ROOT_DIR"):
                value = ENV.get(env_var)
                if value is not None:
                    cmake_cmd.append("-D" + env_var + "=%r" % value)
            log.info("Configuring...")
            if run_process(cmake_cmd) != 0:
                raise DistutilsSetupError(" ".join(cmake_cmd))

            make_cmd = [MAKE_PATH] + MAKE_FLAGS
            log.info("Compiling...")
            if run_process(make_cmd) != 0:
                raise DistutilsSetupError(" ".join(make_cmd))

            make_cmd = [MAKE_PATH, "install"]
            log.info("Installing...")
            if run_process(make_cmd) != 0:
                raise DistutilsSetupError(" ".join(make_cmd))

            if platform.system() == "Darwin":  # macOS
                for filename in DATA_FILES:
                    new_install_name = "@loader_path/" + os.path.basename(filename)
                    install_name_tool_cmd = ["install_name_tool", "-id", new_install_name, filename]
                    log.info("fixing install_name for %s to %r" % (filename, new_install_name))
                    if run_process(install_name_tool_cmd) != 0:
                        raise DistutilsSetupError(" ".join(install_name_tool_cmd))

        # This will generally be called by the manual install
        elif not os.path.isdir(EIGEN3_INCLUDE_DIR):
            raise RuntimeError("Could not find Eigen in EIGEN3_INCLUDE_DIR={}. If doing manual install, please set the EIGEN3_INCLUDE_DIR variable with the absolute path to Eigen manually. If doing install via pip, please file an issue on github.com/clab/dynet".format(EIGEN3_INCLUDE_DIR))

        BUILT_EXTENSIONS = True  # because make calls build_ext
        _build.run(self)


class build_py(_build_py):
    def run(self):
        os.chdir(os.path.join(BUILD_DIR, "python"))
        log.info("Building Python files...")
        _build_py.run(self)


class install_data(_install_data):
    def run(self):
        self.data_files = [(p, f) if self.is_wheel(p) else
                           (get_python_lib(), f) if platform.system() == "Darwin" else
                           (p, []) for p, f in self.data_files]
        _install_data.run(self)

    def is_wheel(self, path):
        return os.path.basename(os.path.abspath(os.path.join(self.install_dir, path))) == "wheel"


class build_ext(_build_ext):
    def run(self):
        if BUILT_EXTENSIONS:
            INCLUDE_DIRS.append(EIGEN3_INCLUDE_DIR)
            LIBRARY_DIRS.append(BUILD_DIR + "/dynet/")
        log.info("Building Cython extensions...")
        log.info("INCLUDE_DIRS=%r" % " ".join(INCLUDE_DIRS))
        log.info("LIBRARIES=%r" % " ".join(LIBRARIES))
        log.info("LIBRARY_DIRS=%r" % " ".join(LIBRARY_DIRS))
        log.info("COMPILER_ARGS=%r" % " ".join(COMPILER_ARGS))
        log.info("EXTRA_LINK_ARGS=%r" % " ".join(EXTRA_LINK_ARGS))
        log.info("RUNTIME_LIB_DIRS=%r" % " ".join(RUNTIME_LIB_DIRS))
        _build_ext.run(self)
        if os.path.abspath(".") != SCRIPT_DIR:
            log.info("Copying built extensions...")
            for d in os.listdir("build"):
                target_dir = os.path.join(SCRIPT_DIR, "build", d)
                rmtree(target_dir, ignore_errors=True)
                try:
                    copytree(os.path.join("build", d), target_dir)
                except OSError as e:
                    log.info("Cannot copy %s %s" % (os.path.join("build",d), e))


try:
    with open(os.path.join(SCRIPT_DIR, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = ""

setup(
    name="dyNET",
    # version="0.0.0",
    install_requires=["cython", "numpy"],
    description="The Dynamic Neural Network Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    author="Graham Neubig",
    author_email="dynet-users@googlegroups.com",
    url="https://github.com/clab/dynet",
    download_url="https://github.com/clab/dynet/releases",
    license="Apache 2.0",
    cmdclass={"build": build, "build_py": build_py, "install_data": install_data, "build_ext": build_ext},
    ext_modules=TARGET,
    py_modules=["dynet", "dynet_viz", "dynet_config"],
    data_files=[(os.path.join("..", ".."), DATA_FILES)],
)
