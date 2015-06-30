from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext = Extension(
        "pycnn",                 # name of extension
        ["pycnn.pyx"],           # filename of our Pyrex/Cython source
        language="c++",              # this causes Pyrex/Cython to create C++ source
        include_dirs=["/home/yogo/Vork/Research/cnn/cnn/",
                      "/home/yogo/Vork/Research/cnn/eigen/"],
        libraries=['cnn'],             # ditto
        library_dirs=["."],
        #extra_link_args=["-L/home/yogo/Vork/Research/cnn/cnn/build/cnn"],       # if needed
        extra_compile_args=["-std=c++11"]

        )

setup(ext_modules = [ext],
        cmdclass = {'build_ext': build_ext}
        )
