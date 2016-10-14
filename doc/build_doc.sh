#!/bin/bash

# Run Doxygen
doxygen doxygen/config

# Run sphinx to generate text and html doc
make html text
