# SWIG bindings for DyNet

The code in `dynet_swig.i` provides SWIG instructions to wrap salient parts of DyNet for use
in other languages, in particular Java.

## Building

To include in the DyNet build, add `-DINCLUDE_SWIG=ON` to the `cmake` command, e.g., run this from
the `build` directory:

```
build$ cmake .. -DEIGEN3_INCLUDE_DIR=../eigen -DINCLUDE_SWIG=ON
build$ make
```

If successful, the end of the build should look something like:

```
[ 96%] Built target dynet_swig
[ 97%] Building Java objects for dynet_swigJNI.jar
[ 98%] Generating CMakeFiles/dynet_swigJNI.dir/java_class_filelist
[100%] Creating Java archive dynet_swigJNI.jar
[100%] Built target dynet_swigJNI
```

and (in MacOS) you should have the library files `build/swig/libdynet_swig.jnilib` and
`build/swig/dynet_swigJNI.jar`.

## Running the example

Here's a simple way to run the example in the `swig/examples` directory:

```
examples$ javac -cp ../../build/swig/dynet_swigJNI.jar XorExample.java
examples$ java -cp .:../../build/swig/dynet_swigJNI.jar -Djava.library.path=../../build/swig XorExample
Running XOR example
[dynet] random seed: 2801594438
[dynet] allocating memory: 512MB
[dynet] memory allocation done.
Dynet initialized!
W = edu.cmu.dynet.Expression@2503dbd3
digraph G {
  rankdir=LR;
  nodesep=.05;
  N0 [label="v0 = parameters({8,2}) @ 0x7fc4d341df20"];
  N1 [label="v1 = parameters({8}) @ 0x7fc4d341e060"];
  N2 [label="v2 = parameters({1,8}) @ 0x7fc4d341e180"];
  N3 [label="v3 = parameters({1}) @ 0x7fc4d341e2f0"];
  N4 [label="v4 = constant({2})"];
  N5 [label="v5 = scalar_constant(0x7fc4d35304b8)"];
  N6 [label="v6 = v0 * v4"];
  N0 -> N6;
  N4 -> N6;
  N7 [label="v7 = v6 + v1"];
  N6 -> N7;
  N1 -> N7;
  N8 [label="v8 = tanh(v7)"];
  N7 -> N8;
  N9 [label="v9 = v2 * v8"];
  N2 -> N9;
  N8 -> N9;
  N10 [label="v10 = v9 + v3"];
  N9 -> N10;
  N3 -> N10;
  N11 [label="v11 = || v10 - v5 ||^2"];
  N10 -> N11;
  N5 -> N11;
}
iter = 0, loss = 3.0038743
iter = 1, loss = 0.011609978
iter = 2, loss = 1.075118E-5
iter = 3, loss = 9.254038E-9
iter = 4, loss = 8.060219E-12
iter = 5, loss = 8.881784E-15
iter = 6, loss = 3.5527137E-15
iter = 7, loss = 3.5527137E-15
iter = 8, loss = 3.5527137E-15
iter = 9, loss = 8.881784E-16
iter = 10, loss = 0.0
iter = 11, loss = 0.0
iter = 12, loss = 0.0
iter = 13, loss = 0.0
iter = 14, loss = 0.0
iter = 15, loss = 0.0
iter = 16, loss = 0.0
iter = 17, loss = 0.0
iter = 18, loss = 0.0
iter = 19, loss = 0.0
iter = 20, loss = 0.0
iter = 21, loss = 0.0
iter = 22, loss = 0.0
iter = 23, loss = 0.0
iter = 24, loss = 0.0
iter = 25, loss = 0.0
iter = 26, loss = 0.0
iter = 27, loss = 0.0
iter = 28, loss = 0.0
iter = 29, loss = 0.0
```