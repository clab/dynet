# C APIs for DyNet

DyNet provides the C APIs that can be used to build bindings for other languages.
The APIs are defined in `dynet_c/api.h`.

## Building the C APIs

To build the C APIs for DyNet, you need to specify `-DENABLE_C=ON` when running `cmake` command
(See the [DyNet documentation](http://dynet.readthedocs.io/en/latest/install.html) for general build instructions).
For example, run this from the `build` directory:

```
build$ cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DENABLE_C=ON
build$ make -j 4
```

To build the C APIs for GPU, make sure you have installed CUDA, then simply add `-DBACKEND=cuda` as a `cmake` option.

## Using the C APIs from other languages

From the `build` directory, run `make install` command to place `dynet_c` library and headers in your installation directory (that is usually `/usr/local`, but you can specify it with `CMAKE_INSTALL_PREFIX`).
When your program or bindings failed to load the library, add the directory holding `libdynet_c.so` to `LD_LIBRARY_PATH`.

Then, the C APIs can be called via the foreign function interface (FFI) in many languages (See the documentation of the language you want to use).

## Usage

The way to use DyNet C APIs is a bit roundabout.
Objects are passed between the C APIs and C++ using pointers.
Because C language has no exception handling, the APIs always return `DYNET_C_STATUS` to let you know whether the API call suceeded or failed.

### Receiving the returned value from C++ DyNet

To receive the value/object from C++ methods/functions, you need to give a pointer when calling the C APIs.

```c
std::uint32_t nd;
DYNET_C_STATUS status = ::dynetGetDimNDimensions(dim, &nd));
// assert(status == DYNET_C_OK);
```

You should pay attention to whether the received object is owned by C++ DyNet.
When C++ methods/functions return an object, it is re-allocated to heap memory and your pointer now holds the address of that object.
In this case what your pointer indicates is not managed by C++ DyNet and you have to delete it when it is no longer needed.

```c
::dynetExpression_t *x;
CHECK_EQUAL(DYNET_C_OK, ::dynetApplyRandomNormal(cg, dim, 0.0, 1.0, &x));
...
CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteExpression(x));
```

In contrast, when C++ methods/functions return an object as reference, your pointer just holds it.
This means the lifetime of what your pointer refers is managed by C++ DyNet, and thus you should not delete that object.

```c
::dynetComputationTensor_t *gradient;
CHECK_EQUAL(DYNET_C_OK, ::dynetGetExpressionGradient(&gradient));

// Do not delete `gradient` because the corresponding method
// `const Tensor& Expression::gradient()` returns reference.
//
// CHECK_EQUAL(DYNET_C_OK, ::dynetDeleteTensor(gradient));
```

### Error handling

When an exception occurs in C++ DyNet, the C APIs always catch it (technically, all `std::exception`s) without aborting your program and then return `DYNET_C_ERROR`.
You should check whether a `DYNET_C_STATUS` value is `DYNET_C_OK` or not wherever your program calls the C APIs.
In case the status is not `DYNET_C_OK`, you can get a message via `dynetGetMessage`.
Error messages that C++ DyNet generates are stored in the thread local storage.
Clear stored messages using `dynetResetStatus()` once you get them.

```c
DYNET_C_STATUS status = ::dynetCreateComputationGraph(nullptr);
if (status != DYNET_C_OK) {
  std::size_t length = 0u;
  ::dynetGetMessage(nullptr, &length);
  char str[length];
  ::dynetGetMessage(str, &length);
  printf("Error message: %s\n", str);
  ::dynetResetStatus();
}
```

## Adding new features to the C APIs

### Adding a new method to the existing class

When a new method is added to the existing class in C++ DyNet, the corresponding C function will need to be added to the C APIs.
For example, if you want a method like `unsigned Tensor::num_elems() const`, you need to define a function like `DYNET_C_STATUS dynetGetTensorNumElems(const dynetTensor_t *tensor, uint32_t *retval)` in `dynet_c/tensor.h` and implement it in `dynet_c/tensor.cc`:

```cpp
DYNET_C_STATUS dynetGetTensorNumElems(
    const dynetTensor_t *tensor, uint32_t *retval) try {
  DYNET_C_CHECK_NOT_NULL(tensor);
  DYNET_C_CHECK_NOT_NULL(retval);
  *retval = to_cpp_ptr(tensor)->num_elems(); // call the C++ method and assign the returned value.
  return DYNET_C_OK;
} DYNET_C_HANDLE_EXCEPTIONS
```

To handle exceptions, wrap the function implementation with the `try { ... } DYNET_C_HANDLE_EXCEPTIONS` block.

### Adding a new class

Objects from C++ DyNet are represented as opaque data types in the C APIs.
If you want to use a new `Foo` C++ class in the C APIs, first define the struct `dynetFoo` in `dynet_c/api.h`.
Defining pointer conversion functions using the macro `DYNET_C_PTR_TO_PTR(Foo, dynetFoo);` is helpful to implement `Foo`'s methods hereafter.
Then, declare the opaque type `dynetFoo_t` in `dynet_c/foo.h`:

```c
typedef struct dynetFoo dynetFoo_t;
```

To implement `Foo`'s methods, see the previous subsection.

### Adding a new file

When you add a new file to the `dynet_c` directory, add its filename to `dynet_c/CMakeLists.txt` and `dynet_c/api.h`.

### Naming Conventions

We use `dynet` as a prefix for all structs and functions because C language has no namespace.
When you add a new function, name it in the form like `dynetVerbHandlerObjective`.

## Acknowledgements

The DyNet C APIs are designed in reference to [primitiv](https://github.com/primitiv/primitiv).

