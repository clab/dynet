# SWIG bindings for DyNet in Scala/Java

The code in `dynet_swig.i` provides SWIG instructions to wrap salient
parts of DyNet for use in other languages, in particular Scala (and
Java). It produces Java (in `edu.cmu.dynet.internal`) that slavishly
recreates the C++ API, and that you should strive not to use.

Instead you should use the Scala API that lives in `edu.cmu.dynet`.
It recreates all the functionality of the C++ version, but is designed
to be idiomatic Scala.

There are many examples in `edu/cmu/dynet/examples` that illustrate
how to build / train / use models.

## Building the Scala Bindings

You need to have a recent version of SWIG installed (3.0.11 or later),
which you can download from [swig.org](http://www.swig.org/).
Note that if you are using Ubuntu, `apt-get` will almost certainly install
a much older version that won't work here. You also need to make sure that
your `$JAVA_HOME` environment variable is set correctly.

Then to build DyNet with the SWIG bindings, simply add `-DINCLUDE_SWIG=ON` to the
`cmake` command. (See the [DyNet
documentation](http://dynet.readthedocs.io/en/latest/install.html) for
general build instructions). For example, run this from the `build`
directory:

```
build$ cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DINCLUDE_SWIG=ON
build$ make
```

If successful, the end of the build will look like:

```
[ 95%] Built target dynet_swig
[ 96%] Building Java objects for dynet_swigJNI.jar
[ 97%] Generating CMakeFiles/dynet_swigJNI.dir/java_class_filelist
[ 98%] Creating Java archive dynet_swigJNI.jar
[ 98%] Built target dynet_swigJNI
updating: libdynet_swig.jnilib (deflated 83%)
[ 98%] Built target dylib_into_jar
[100%] Running sbt
...
[info] lots of sbt stuff here
...
[success] Total time: 14 s, completed Mar 1, 2017 11:38:54 AM
[100%] Built target scala_helper
```

This command runs SWIG to generate a dynamic library file and JNI
bindings, then runs `sbt assembly` to produce an "uberjar" containing
both the Dynet bindings and the Scala helpers. It outputs three
artifacts in the `build/swig` directory: `dynet_swigJNI.jar`,
`dynet_swigJNI_dylib.jar` and `dynet_swigJNI_scala.jar`. Note that
`dynet_swigJNI_dylib.jar` contains a native library that is not
portable across systems.

To include DyNet in a Scala project, add both
`dynet_swigJNI_dylib.jar` and `dynet_swigJNI_scala.jar` to your
classpath. If you're using `sbt`, you can put these in a `lib/`
directory under the project root directory. To include DyNet in a Java
project, add both `dynet_swigJNI_dylib.jar` and `dynet_swigJNI.jar` to
your classpath.

### Disabling the Scala API

If you don't want the Scala API (and, in particular, if you
don't have `sbt`) then when you run `cmake` include the additional flag

```
-DINCLUDE_SCALA=OFF
```

But really you want the Scala API.

### Building for GPU

To build the GPU version, make sure you have CUDA installed, then
simply add `-DBACKEND=cuda` as a `cmake` option. The generated `.jar`
and dynamic library will use the GPU for computations. (Currently
there's no way to build a version that can do both CPU and GPU, you
would have to build two separate versions if you wanted that.)

The CPU and GPU versions of DyNet behave almost identically. The one
difference is that the `DynetParams` class has additional fields in
the GPU version for configuring the GPU. This difference shouldn't
affect you unless you want to configure these parameters and also run
the same code on the CPU.

## Running the Examples

Several examples using DyNet from Scala and Java are included under
`src/main/[scala|java]/edu/cmu/dynet/examples/`. After running `make`,
you can run the Scala examples using `sbt` from the `swig` directory:

```
swig$ sbt "runMain edu.cmu.dynet.examples.XorScala"
```

The Java example takes a couple more steps:

```
swig$ javac -d . -cp ../build/swig/dynet_swigJNI.jar src/main/java/edu/cmu/dynet/examples/XorExample.java
swig$ java -cp .:../build/swig/dynet_swigJNI.jar:../build/swig/dynet_swigJNI_dylib.jar edu.cmu.dynet.examples.XorExample
```

In both cases, you should see output like:

```
Running XOR example
[dynet] random seed: 1650744221
[dynet] allocating memory: 512MB
[dynet] memory allocation done.
Dynet initialized!

Computation graphviz structure:
digraph G {
  rankdir=LR;
  nodesep=.05;
  N0 [label="v0 = parameters({8,2}) @ 0x7ff1da8000e0"];
  N1 [label="v1 = parameters({8}) @ 0x7ff1da800260"];
  N2 [label="v2 = parameters({1,8}) @ 0x7ff1da800380"];
  N3 [label="v3 = parameters({1}) @ 0x7ff1da8004f0"];
  N4 [label="v4 = constant({2})"];
  N5 [label="v5 = scalar_constant(0x7ff1d8f12ce8)"];
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

Training...
iter = 0, loss = 0.6974922
iter = 1, loss = 1.4101544E-4
iter = 2, loss = 2.1905963E-8
iter = 3, loss = 3.2875924E-12
iter = 4, loss = 2.220446E-15
iter = 5, loss = 8.881784E-16
iter = 6, loss = 8.881784E-16
iter = 7, loss = 8.881784E-16
iter = 8, loss = 8.881784E-16
iter = 9, loss = 8.881784E-16
iter = 10, loss = 8.881784E-16
iter = 11, loss = 8.881784E-16
iter = 12, loss = 8.881784E-16
iter = 13, loss = 8.881784E-16
iter = 14, loss = 8.881784E-16
iter = 15, loss = 8.881784E-16
iter = 16, loss = 8.881784E-16
iter = 17, loss = 8.881784E-16
iter = 18, loss = 8.881784E-16
iter = 19, loss = 8.881784E-16
iter = 20, loss = 8.881784E-16
iter = 21, loss = 8.881784E-16
iter = 22, loss = 8.881784E-16
iter = 23, loss = 8.881784E-16
iter = 24, loss = 8.881784E-16
iter = 25, loss = 8.881784E-16
iter = 26, loss = 8.881784E-16
iter = 27, loss = 8.881784E-16
iter = 28, loss = 8.881784E-16
iter = 29, loss = 8.881784E-16
```

## Usage

The current Scala API works mostly like the C++ API, with the following
differences.

### Naming

Everything has been given Scala-cased names. So `affine_transform`
becomes `affineTransform` and so on. 

### `ComputationGraph`s

In Scala there is a singleton `ComputationGraph`. Accordingly, any 
function or method that in C++ would take the computation graph as a 
parameter, in Scala doesn't. 

When you want to clear the computation graph and get a new one, call the
static method

```scala
ComputationGraph.renew()
```

Any `Expression` instances that were associated with previous computation
graphs will become stale, and you'll get an error if you try to use them.

All of the C++ `ComputationGraph` instance methods are in Scala static methods
on the `ComputationGraph` companion object.

### `Expression` functions

The C++ API defines a large number of bare functions for creating `Expression`s.
In Scala these are all static methods on the `Expression` companion object.

### `std::vector`s

DyNet does a lot behind the scenes with C++ `std::vector<>`s. 
In Scala there are `IntVector`, `FloatVector`, `UnsignedVector`, and
`ExpressionVector` classes that thinly wrap these C++ vectors. 
They all implement `IndexedSeq` so that they're pretty easy to work
with.

Each has a `size: Int` constructor and a `values: Seq[_]` constructor.

### unsigned ints

SWIG converts C++ `unsigned` variables to Java `Long` variables.
This results in some minor unpleasantness where (for example)
any function that takes a `std::vector<unsigned>` on the C++ side
takes an `UnsignedVector` on the Scala side, but any function that
takes an `unsigned` on the C++ side takes a `Long` on the C++ side.

### `Vector` scope

The `FloatVector` class is a wrapper around a native C++ `vector<float>`. 
Some of the `input` functions take a `FloatVector` parameter, and behind
the scenes they are copying a _pointer_ to the underlying C++ vector.

Whenever the Scala `FloatVector` gets garbage-collected, it will delete
the underlying C++ vector. If you still have computations that point to
it, you will get a segfault.

Therefore, in this situation it is *very important* to keep the 
`FloatVector`s in scope so that they don't get prematurely garbage-collected.

For example, imagine you have a function:

```scala
def getImage(index: Int): FloatVector
```

that reads an image off disk.

The following is bad:

```scala
// THIS IS BAD
for (idx <- indexes) {
  val x = Expression.input(Dim(100), getImage(idx))
  // some other computations
  ComputationGraph.forward(loss_expr)
  ComputationGraph.backward(loss_expr)
}
```

as the FloatVector that comes back from `getImage` is eligible (and likely)
to be garbage collected before `forward` (which runs a chain of computations
that involve a pointer to the deleted vector) ever runs.

Instead, you need to do something like

```scala
// THIS IS FINE
for (idx <- indexes) {
  val image = getImage(idx)
  val x = Expression.input(Dim(100), image)
  // some other computations
  ComputationGraph.forward(loss_expr)
  ComputationGraph.backward(loss_expr)
}
```

because now `image` can't be garbage collected until after
its iteration is finished.

(It is possible, even likely, that there are other ways in which this
 phenomenon might manifest, so keep an eye out.)

### Pointers

For the most part you shouldn't have to worry about pointers; most things
are references in Java, so where a C++ method might take a 
`Model*` parameter, the corresponding Java method just takes a `Model`.

The exception is primitives. SWIG produces (for example) a `SWIGTYPE_p_int`
type wrapper for `int*` and bare functions for working with these. 
The Scala API provides wrapper classes `IntPointer` and `FloatPointer`
that are nicer to work with and that implicitly convert to the SWIG types.

### Serialization

DyNet uses `boost::serialization` to correctly serialize/deserialize
an object graph of `Model`s, `Builder`s, and so on. On the Java side
we expose `ModelLoader` and `ModelSaver` classes that wrap this
functionality and allow complex serialization from Scala code. We also
extended these classes to serialize primitives and Java/Scala objects
that implement `Serializable`:

```scala
    val mod1 = new Model()
    val rnn1 = new SimpleRNNBuilder(1, 10, 10, mod1)

    val path = "/path/to/save/model/files"
    val saver = new ModelSaver(path)
    saver.addModel(mod1)
    saver.addSRNNBuilder(rnn1)
    saver.addObject(new Foo())
    saver.addInt(3)
    saver.done()

    val loader = new ModelLoader(path)
    val mod2 = loader.loadModel()
    val rnn2 = loader.loadSRNNBuilder()
    val foo = loader.loadObject(classOf[Foo])
    val i = loader.loadInt()
    loader.done()
```

The `ModelSaver` doesn't do any tracking of what it saves (or in what order),
so it's on you to track that and/or make sure you deserialize things in the
same order they were serialized.

