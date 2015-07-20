from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
       cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

ext = Extension(
        "pycnn",                 # name of extension
        ["pycnn.pyx"],           # filename of our Pyrex/Cython source
        language="c++",              # this causes Pyrex/Cython to create C++ source
        include_dirs=["../../cnn/",
                      "../../eigen/"],
        libraries=['cnn_shared'],             # ditto
        library_dirs=["."],
        #extra_link_args=["-L/home/yogo/Vork/Research/cnn/cnn/build/cnn"],       # if needed
        extra_compile_args=["-std=c++11"],
        runtime_library_dirs=["$ORIGIN/./"],

        )

setup(ext_modules = [ext],
        cmdclass = {'build_ext': build_ext}
        )
