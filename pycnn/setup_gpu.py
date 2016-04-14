from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext


# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
       cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

ext = Extension(
        "gpycnn",                 # name of extension
        ["gpycnn.pyx"],           # filename of our Pyrex/Cython source
        language="c++",              # this causes Pyrex/Cython to create C++ source
        include_dirs=["../../cnn/", # this is the location of the main cnn directory.
                      "../../eigen/"], # this is the directory where eigen is saved.
        #libraries=['cnn','cnncuda'], #,'cnncuda_shared'],             # ditto
        libraries=['gcnn_shared','cnncuda_shared'],             # ditto
        library_dirs=["."],
        #extra_link_args=["-L/home/yogo/Vork/Research/cnn/cnn/build/cnn"],       # if needed
        extra_compile_args=["-std=c++11","-fPIC"],#,"-lcudart","-lcublas"],
		extra_link_args=["-L/usr/local/cuda-7.5/lib64","-lcudart","-lcublas"],
		#extra_objects=["libcnncuda.a"],
        )

setup(ext_modules = [ext],
        cmdclass = {'build_ext': build_ext},
        name="gpyCNN",
        )
