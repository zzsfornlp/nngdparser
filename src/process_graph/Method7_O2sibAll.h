/*
 * Method7_O2sibAll.h
 *
 *  Created on: 2015Äê3ÔÂ15ÈÕ
 *      Author: zzs
 */

#ifndef PROCESS_METHOD7_O2SIBALL_H_
#define PROCESS_METHOD7_O2SIBALL_H_

#include "Process.h"
//method7: o2sib

class Method7_O2sibAll: public Process{
protected:
	REAL* data;
	int current;
	int end;
	REAL* gradient;
	NNInterface * mach_o1;
	REAL* target;
public:
	Method7_O2sibAll(string conf):Process(conf){
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
	virtual REAL* each_next_data(int*);
	virtual void each_get_grad(int);
	virtual void init_embed();
	virtual int each_get_mach_outdim(){return 2;}

	virtual void each_get_featgen(int if_testing){
		if(if_testing){
			if(! feat_gen)	//when testing
				feat_gen = new FeatureGenO2sib(dict,parameters->CONF_x_window,parameters->CONF_add_distance,
						parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
			feat_gen->deal_with_corpus(dev_test_corpus);
		}
		else{
			feat_gen = new FeatureGenO2sib(dict,parameters->CONF_x_window,parameters->CONF_add_distance,
					parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
			feat_gen->deal_with_corpus(training_corpus);
		}
	}
	virtual vector<int>* each_test_one(DependencyInstance* x);
};


#endif /* PROCESS_METHOD7_O2SIBALL_H_ */
