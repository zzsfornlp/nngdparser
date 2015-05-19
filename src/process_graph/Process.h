/*
 * Process.h
 *
 *  Created on: 2015Äê3ÔÂ18ÈÕ
 *      Author: zzs
 */

#ifndef PROCESS_GRAPH_PROCESS_H_
#define PROCESS_GRAPH_PROCESS_H_

#include "../parts/Parameters.h"
#include "../parts/FeatureGen.h"
#include "../parts/FeatureGenO1.h"
#include "../parts/FeatureGenO2sib.h"
#include "../parts/Dict.h"
#include "../tools/CONLLReader.h"
#include "../tools/DependencyEvaluator.h"
#include "../nn/NNInterface.h"
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
using namespace std;

#define GET_MAX_ONE(a,b) (((a)>(b))?(a):(b))
#define GET_MIN_ONE(a,b) (((a)>(b))?(b):(a))

class Process{
protected:
	//data
	parsing_conf* parameters;
	FeatureGen* feat_gen;
	Dict* dict;
	vector<DependencyInstance*>* training_corpus;
	vector<DependencyInstance*>* dev_test_corpus;

	//read from restart file or from scratch
	REAL cur_lrate;
	int cur_iter;
	int CTL_continue;	//if continue training
	double * dev_results;	//the results of dev-data
	int lrate_cut_times;	//number of times of lrate cut
	NNInterface *mach;
	//init embedings	--- here for convenience(should be put to NNInterface)
	virtual void init_embed();

	//some procedures
	void set_lrate();					//no schedule, just decrease lrate
	int set_lrate_one_iter();	//lrate schedule
	virtual int whether_keep_trainning();
	//restart files
	void read_restart_conf();
	void write_restart_conf();
	void delete_restart_conf();

	//help
	static void shuffle_data(REAL* x,REAL* y,int xs,int ys,int xall,int yall,int times);
	static void set_softmax_gradient(const REAL* s_target,const REAL* s_output,REAL* s_gradient,int bsize,int c);
	static void set_pair_gradient(const REAL* s_output,REAL* s_gradient,int bsize);
	//parse
	static double* get_scores_o1(DependencyInstance* x,parsing_conf* zp,NNInterface * zm,FeatureGen* zf);		//double[l*l]
	static double* get_scores_o2sib(DependencyInstance* x,parsing_conf* zp,NNInterface * zm,FeatureGen* zf,bool* score_o1=0);	//double[l*l*l]
	static double* get_scores_o2g(DependencyInstance* x,parsing_conf* zp,NNInterface * zm,FeatureGen* zf,bool* score_o1=0);	//double[l*l*l]
	static double* get_scores_o3g(DependencyInstance* x,parsing_conf* zp,NNInterface * zm,FeatureGen* zf,bool* score_o1);	//double[l*l*l*l]
	vector<int>* parse_o1(DependencyInstance* x);
	vector<int>* parse_o2sib(DependencyInstance* x,double* score_of_o1=0);
	vector<int>* parse_o2g(DependencyInstance* x,double* score_of_o1=0);
	vector<int>* parse_o3g(DependencyInstance* x,double* score_of_o1,double* score_of_o2sib,double* score_of_o2g);
	int score_noprob(double score){
		//impossible scores-prob for o2-o1-pruning
		return score < parameters->CONF_NN_highO_o1filter_cut;
	}

	//train and test
	double nn_dev_test(string to_test,string output,string gold);
	void nn_train_one_iter();
	void nn_train_prepare();

	//virtual functions for different methods
	virtual int each_get_mach_outdim(){return 2;}	//really bad design --- default 2(binary)
	virtual void each_prepare_data_oneiter()=0;
	virtual REAL* each_next_data(int*)=0;
	virtual void each_get_grad(int)=0;
	virtual vector<int>* each_test_one(DependencyInstance* x){
		//default o1
		return parse_o1(x);
	}
	virtual void each_get_featgen(int if_testing){
		// default only the order1 features
		if(if_testing){
			if(! feat_gen){	//when testing
				feat_gen = new FeatureGenO1(dict,parameters->CONF_x_window,parameters->CONF_add_distance,
						parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
			}
			feat_gen->deal_with_corpus(dev_test_corpus);
		}
		else{
			feat_gen = new FeatureGenO1(dict,parameters->CONF_x_window,parameters->CONF_add_distance,
					parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
			feat_gen->deal_with_corpus(training_corpus);
		}
	}

public:
	Process(string);
	virtual ~Process(){}
	virtual void train();
	virtual void test(string);
	//special checking for o1-filter
	void check_o1_filter(string,string);
};


#endif /* PROCESS_GRAPH_PROCESS_H_ */
