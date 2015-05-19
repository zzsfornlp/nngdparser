/*
 * Method9_O3g.h
 *
 *  Created on: 2015Äê4ÔÂ23ÈÕ
 *      Author: zzs
 */

#ifndef PROCESS_GRAPH_METHOD9_O3G_H_
#define PROCESS_GRAPH_METHOD9_O3G_H_

#include "Process.h"
#include "../parts/FeatureGenO2sib.h"
#include "../parts/FeatureGenO2g.h"
#include "../parts/FeatureGenO3g.h"

//Method9: o3grand-sibling(maybe no gsib-score)

class Method9_O3g: public Process{
protected:
	REAL* data;
	REAL* gradient;
	REAL* target;
	int current;
	int end;
	//other machines
	NNInterface * mach_o1;
	NNInterface * mach_o2sib;
	NNInterface * mach_o2g;
public:
	Method9_O3g(string conf):Process(conf){
		current = end = 0;
		data = 0;
		gradient = 0;
		target = 0;
		mach_o1 = mach_o2sib = mach_o2g = 0;
		if(parameters->CONF_NN_highO_o1mach.length() > 0)
			mach_o1 = NNInterface::Read(parameters->CONF_NN_highO_o1mach);
		if(parameters->CONF_NN_highO_o2sibmach.length() > 0)
			mach_o2sib = NNInterface::Read(parameters->CONF_NN_highO_o2sibmach);
		if(parameters->CONF_NN_highO_o2gmach.length() > 0)
			mach_o2g = NNInterface::Read(parameters->CONF_NN_highO_o2gmach);
	}
	virtual void each_prepare_data_oneiter();
	virtual void each_get_grad(int);
	virtual REAL* each_next_data(int*);
	virtual int each_get_mach_outdim(){return 2;}
	vector<int>* each_test_one(DependencyInstance* x);
	virtual void each_get_featgen(int if_testing){
		if(if_testing){
			if(! feat_gen)	//when testing
				feat_gen = new FeatureGenO3g(dict,parameters->CONF_x_window,parameters->CONF_add_distance,
						parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
			feat_gen->deal_with_corpus(dev_test_corpus);
		}
		else{
			feat_gen = new FeatureGenO3g(dict,parameters->CONF_x_window,parameters->CONF_add_distance,
					parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
			feat_gen->deal_with_corpus(training_corpus);
		}
	}
};


#endif /* PROCESS_GRAPH_METHOD9_O3G_H_ */
