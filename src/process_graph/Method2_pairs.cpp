/*
 * Method2_pairs.cpp
 *
 *  Created on: 8 Jan, 2015
 *      Author: z
 */

#include "Method2_pairs.h"
void Method2_pairs::each_prepare_data_oneiter()
{
	delete []data;
	delete []gradient;

	//for gradient
	gradient = new REAL[mach->GetWidth()*mach->GetOdim()];
	mach->SetGradOut(gradient);

	//prepare all
	//-- first all
	int num_pairs = 0;
	int sentences = training_corpus->size();
	for(int i=0;i<sentences;i++){
		int length = training_corpus->at(i)->length();
		//here duplicate right ones and exclude root as mod
		// -- length-2 excludes self,real-head
		num_pairs += (length-2)*(length-1)*2;
	}
	//-- generate all
	int real_num_pairs = 0;
	data = new REAL[num_pairs*mach->GetIdim()];
	REAL* assign_x = data;
	FeatureGenO1* feat_o1 = (FeatureGenO1*)feat_gen;	//force it
	for(int i=0;i<sentences;i++){
		DependencyInstance* x = training_corpus->at(i);
		int length = x->length();
		for(int mod=1;mod<length;mod++){
			int head = x->heads->at(mod);
			for(int j=0;j<length;j++){	//length-2
				if(j==head || j==mod)
					continue;
				//always first right and then wrong
				feat_gen->fill_one(assign_x,x,head,mod);
				assign_x += mach->GetIdim();
				feat_gen->fill_one(assign_x,x,j,mod);
				assign_x += mach->GetIdim();
				real_num_pairs += 2;
			}
		}
	}
	current = 0;
	end = real_num_pairs;
	//shuffle --- make sure shuffle 2 at the same time(here really lazy to write another shuffle,so ...)
	shuffle_data(data,data,2*mach->GetIdim(),2*mach->GetIdim(),
			real_num_pairs*mach->GetIdim(),real_num_pairs*mach->GetIdim(),10);
	//sample
	cout << "--Data for this iter: samples all " << end << " resample: " << (int)(end*parameters->CONF_NN_resample) << endl;
	end = (int)(end*parameters->CONF_NN_resample);
}

REAL* Method2_pairs::each_next_data(int* size)
{
	//size must be even number, if no universal rays, it should be that...
	if(current >= end)
		return 0;
	if(current + *size > end)
		*size = end - current;
	//!!not adding current here
	return (data+current*mach->GetIdim());
}

void Method2_pairs::each_get_grad(int size)
{
	set_pair_gradient(mach->GetDataOut(),gradient,size);
	//!! here add it
	current += size;
}

