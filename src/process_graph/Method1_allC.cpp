/*
 * Method1_allC.cpp
 *
 *  Created on: Dec 25, 2014
 *      Author: zzs
 */

#include "Method1_allC.h"
void Method1_allC::each_prepare_data_oneiter()
{
	delete []data;
	delete []target;
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
		num_pairs += length*(length-1);
	}
	//-- generate all
	int real_num_pairs = 0;
	data = new REAL[num_pairs*mach->GetIdim()];
	target = new REAL[num_pairs];
	REAL* assign_x = data;
	REAL* assign_y = target;
	FeatureGenO1* feat_o1 = (FeatureGenO1*)feat_gen;	//force it
	for(int i=0;i<sentences;i++){
		DependencyInstance* x = training_corpus->at(i);
		int length = x->length();
		for(int m=1;m<length;m++){
			for(int h=0;h<length;h++){
				if(h != m){
					//build mach_x
					REAL t = 0;
					feat_gen->fill_one(assign_x,x,h,m);
					if(x->heads->at(m)==h)
						t=1;
					*assign_y = t;
					assign_x += mach->GetIdim();
					assign_y += 1;
					real_num_pairs++;
				}
			}
		}
	}
	current = 0;
	end = real_num_pairs;
	//shuffle
	shuffle_data(data,target,mach->GetIdim(),1,real_num_pairs*mach->GetIdim(),real_num_pairs,10);
	//sample
	cout << "--Data for this iter: samples all " << end << " resample: " << (int)(end*parameters->CONF_NN_resample) << endl;
	end = (int)(end*parameters->CONF_NN_resample);
}

REAL* Method1_allC::each_next_data(int* size)
{
	if(current >= end)
		return 0;
	if(current + *size > end)
		*size = end - current;
	//!!not adding current here
	return (data+current*mach->GetIdim());
}

void Method1_allC::each_get_grad(int size)
{
	set_softmax_gradient(target+current,mach->GetDataOut(),gradient,size,mach->GetOdim());
	//!! here add it
	current += size;
}
