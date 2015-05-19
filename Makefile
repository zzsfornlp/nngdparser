## the makefile for nn-graph-dparser(nngdparser)
#to compile the parser, we need boost_regex boost_program_options and blas library

CPP=g++
CC=gcc
CFLAGS=-O3 
LFLAGS=-O3
LD=g++

#!!!specify the blas lib location and the blas lib, these may not be the same in different machines
BLAS_LIBS_LOCATION=-L/usr/lib/atlas-blas
BLAS_LIBS=-lf77blas  

###choose the blas implementation
BLAS_DEFINE=-DBLAS_ATLAS
#BLAS_DEFINE=-DBLAS_INTEL_MKL

SRCS=$(wildcard src/*/*.cpp) src/main.cpp
#no source files with the same name
OBJS=$(patsubst %.cpp,obj/%.o,$(notdir $(SRCS))) obj/Blas.o

nngdparser: obj/depends $(OBJS)
	$(LD) $(LFLAGS) $(OBJS) -o nngdparser $(BLAS_LIBS_LOCATION) -lboost_regex -lboost_program_options $(BLAS_LIBS)

obj/depends:
	$(CPP) $(CFLAGS) $(BLAS_DEFINE) -MM $(SRCS) > $@
    
obj/%.o: src/%.cpp
	$(CPP) $(CFLAGS) $(BLAS_DEFINE) -c $< -o $@
obj/%.o: src/algorithms/%.cpp
	$(CPP) $(CFLAGS) $(BLAS_DEFINE) -c $< -o $@
obj/%.o: src/cslm/%.cpp
	$(CPP) $(CFLAGS) $(BLAS_DEFINE) -c $< -o $@
obj/%.o: src/nn/%.cpp
	$(CPP) $(CFLAGS) $(BLAS_DEFINE) -c $< -o $@
obj/%.o: src/parts/%.cpp
	$(CPP) $(CFLAGS) $(BLAS_DEFINE) -c $< -o $@
obj/%.o: src/process_graph/%.cpp
	$(CPP) $(CFLAGS) $(BLAS_DEFINE) -c $< -o $@
obj/%.o: src/tools/%.cpp
	$(CPP) $(CFLAGS) $(BLAS_DEFINE) -c $< -o $@
obj/Blas.o: src/cslm/Blas.c
	$(CC) $(CFLAGS) -c $< -o $@

include obj/depends

.PHONY: clean
clean:
	rm -f obj/*.o nngdparser
    
