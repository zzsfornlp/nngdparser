/*
 * Method8_O2g.h
 *
 *  Created on: 2015Äê4ÔÂ21ÈÕ
 *      Author: zzs
 */

#ifndef PROCESS_GRAPH_METHOD8_O2G_H_
#define PROCESS_GRAPH_METHOD8_O2G_H_

#include "Process.h"
#include "../parts/FeatureGenO2g.h"
//method8: o2g --- grandchild
// -- kind of like M6

class Method8_O2g: public Process{
protected:
	REAL* data;
	int current;
	int end;
	REAL* gradient;
	NNInterface * mach_o1;
	REAL* target;
public:
	Method8_O2g(string conf):Process(conf){
		current = end = 0;
		data = 0;
		gradient = 0;
		target = 0;
		if(parameters->CONF_NN_highO_o1mach.length() > 0){
			mach_o1 = NNInterface::Read(parameters->CONF_NN_highO_o1mach);
		}
		else
			mach_o1 = 0;
	}
	virtual void each_prepare_data_oneiter();
	virtual void each_get_grad(int);
	virtual REAL* each_next_data(int*);
	virtual int each_get_mach_outdim(){return 2;}
	vector<int>* each_test_one(DependencyInstance* x);

	virtual void each_get_featgen(int if_testing){
		if(if_testing){
			if(! feat_gen)	//when testing
				feat_gen = new FeatureGenO2g(dict,parameters->CONF_x_window,parameters->CONF_add_distance,
						parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
			feat_gen->deal_with_corpus(dev_test_corpus);
		}
		else{
			feat_gen = new FeatureGenO2g(dict,parameters->CONF_x_window,parameters->CONF_add_distance,
					parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
			feat_gen->deal_with_corpus(training_corpus);
		}
	}
};



#endif /* PROCESS_GRAPH_METHOD8_O2G_H_ */
