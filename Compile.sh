#!/bin/bash

##compile all of it once, again the libaries and their locations may be different on different machines...

gcc -O3 -c src/cslm/Blas.c
g++ -O3 -DBLAS_ATLAS  Blas.o src/*.cpp src/*/*.cpp -L/usr/lib/atlas-base -lboost_regex -lboost_program_options -lf77blas -o nngdparser
rm Blas.o

