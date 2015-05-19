nngdparser --- neural network graph-based dependency parser

========================================
SOME NOTES ABOUT IMPLEMENTATION:
Sources:
src/algorithms: 
    The parsing algorithms for graph-based dependency parsing, which are Einser's(1996) algorithm and its extensions to high-order.
    Here we only implement the 1-best tree version for o1,o2sibling,o2grandchildren,o3grandsibling, for k-best extensions, k-best info should be add to the charts.
    
src/cslm:
    Here is the neural network parts, here we use the nn machine parts of the cslm toolkit(http://www-lium.univ-lemans.fr/cslm/).
    
src/nn:
    We also add another interface layer as Decorator above the neural networks to realize some changes for cslm (we use this to add pre-calculation for cslm without changing cslm's codes).
    More importantly, in future versions, we may choose some neural networks tools other than cslm; for now, we only use cslm toolkit for the nn part.

src/parts:
    some important parts for the parsing, including: 
    1.dictionary which record the words' indexes; 
    2.the so-called feature-generators which generate the inputs for neural networks
        --- unfortunately, the indexing scheme of feature-generators are different than those of the parsing algorithms, sorry about this bad design.
    3.the parameters class (the whole program's options)
    
src/process_graph:
    The main process part for the parser, the Process class's methods are the TEMPLATE methods, and those Method* class are for different parsing algorithms.
    Method2 is a method for pairwise training using max(0,1-f(right)+f(wrong)) as object function, this works well for order1, but not for higher orders; other methods can be judged by their names.
        --- For historical reasons, the indexes in the names may be strange, please don't mind.
        
src/tools:
    some processing tools, such as input-output for conll-format and Evaluators of UAS.
    Most of these parts are from the MaxParser(http://sourceforge.net/projects/maxparser/), which is is written in c++ and can parsing with first, second, third and fourth order projective Dependency Parsing Algorithm.
    
Libraries:
    For cslm, we should include C++ boost library and blas library(we use atlas-blas, the makefile should be modified if using other blas implementations), please check out the makefile and the compiling script.
================================================