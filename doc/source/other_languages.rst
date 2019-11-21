Installing/Using in Other Languages
===================================

DyNet mainly supports the C++ and Python bindings, but there are also bindings for
other languages that have been contributed by the community.

C APIs
----------

DyNet provides the `C APIs <https://github.com/clab/dynet/tree/master/contrib/c>`_ that can be used to build bindings for other languages. Please see the README linked above for details.

Rust
----------

DyNet has `Rust Bindings <https://github.com/clab/dynet/tree/master/contrib/rust>`_
developed by Hiroki Teranishi at Nara Institute of Science and Technology. Please see
the README linked above for details.

Scala/Java
----------

DyNet has `Scala/Java Bindings <https://github.com/clab/dynet/tree/master/contrib/swig>`_
developed by Joel Grus at the Allen Institute for Artificial Intelligence. Please see
the README linked above for details.

The `CLU Lab <http://clulab.cs.arizona.edu/>`_ at the University of Arizona has packaged
the Scala/Java bindings into a single, multi-platform jar file that can be incorporated
into a project as a simple library dependency.  This
`fatdynet <https://github.com/clulab/fatdynet>`_ is also able to read models directly
from jar or zip files so that they can be similarly deployed.
