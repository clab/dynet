#!/bin/bash

# Run Doxygen
cd doxygen
doxygen
cd ..

# Run sphinx to generate text and html doc
make html
