## nngdparser: Graph-based Dependency Parser with Neural Netwrok ##
------------------------------------------------------

### Intro ###

This repo contains the implementation for a graph-based dependency parser with neural network. The parser is written in c++, and use simple feed-forward neural network models for high-order graph-based dependency parsing (with order1, order2-sibling, order2-grandchildren and order3-grandsibling).

Please check the **updated version** of our parser [nnpgdparser](https://github.com/zzsfornlp/nnpgdparser).

### How to compile ###

Change to the top-layer directory of the project, and Run "make" or directly run "bash Compile.sh" for one-time compiling.

The "nngdparser" is the runnable file for the parser.

This is the environment where we compile it, if you are interested in compiling in other environments, please figure out the library dependents.

	Platform: Linux os
	Compiler: g++, gcc
	Libraries: Boost C++ libraries, Blas (atlas or mkl).

For more informations of compiling and the libraries, please check out the makefile or the compile script.

### How to run ###

For the training and testing part, please check out the `doc/Usage.txt` file for details.

### Related paper ###

This is the implementation of our paper in Paclic-29. 

	@InProceedings{zzs2015,
		author    = {Zhang, Zhisong and Zhao, Hai},
		title     = {High-order Graph-based Neural Dependency Parsing},
		booktitle = {Proceedings of the 29th Pacific Asia Conference on Language, Information, and Computation},
		month     = {October},
		year      = {2015},
		pages     = {114-123},
		address   = {Shanghai, China},
	}  
