/*
 * Method7_O2sibAll.cpp
 *
 *  Created on: 2015Äê3ÔÂ15ÈÕ
 *      Author: zzs
 */

#include "Method7_O2sibAll.h"
#include "../algorithms/Eisner.h"
#include <cstdio>

void Method7_O2sibAll::each_prepare_data_oneiter()
{
	delete []data;
	delete []target;
	delete []gradient;
	//for gradient
	gradient = new REAL[mach->GetWidth()*mach->GetOdim()];
	mach->SetGradOut(gradient);
	FeatureGenO2sib* feat_o2 = (FeatureGenO2sib*)feat_gen;	//force it
	int sentences = training_corpus->size();
	int idim = mach->GetIdim();
	int odim = mach->GetOdim();

	//only one time when o1_filter(decoding o1 is quite expensive)
	static REAL* data_right = 0;
	static REAL* data_wrong = 0;
	static int tmpall_right=0;
	static int tmpall_wrong=0;
	static int tmpall_bad=0;
	int whether_o1_filter = 0;
	if(parameters->CONF_NN_highO_o1mach.length() > 0 && parameters->CONF_NN_highO_o1filter)
		whether_o1_filter = 1;

	if(data_right==0 || !whether_o1_filter){
	//sweep all once and count
	FeatureGenO1* feat_temp_o1 = new FeatureGenO1(dict,parameters->CONF_x_window,
					parameters->CONF_add_distance,parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
	double** all_scores_o1 = new double*[sentences];
	int all_tokens_train=0,all_token_filter_wrong=0;
	for(int i=0;i<sentences;i++){
		all_scores_o1[i] = 0;
		if(whether_o1_filter){
			DependencyInstance* x = training_corpus->at(i);
			all_scores_o1[i] = get_scores_o1(x,parameters,mach_o1,feat_temp_o1);
			double* scores_o1_filter = all_scores_o1[i];
			all_tokens_train += x->length();
			for(int i2=1;i2<x->length();i2++){	//ignore root
				if(score_noprob(scores_o1_filter[get_index2(x->length(),x->heads->at(i2),i2)]))
					all_token_filter_wrong ++;
			}
		}
	}
	if(whether_o1_filter)
		cout << "For o1 filter: all " << all_tokens_train << ";filter wrong " << all_token_filter_wrong << endl;
	time_t now;
	time(&now);cout << "#Finish o1-filter at " << ctime(&now) << endl;
	//************WE MUST SPECIFY O1_FILTER****************//
	if(!whether_o1_filter){
		cout << "No o1-filter for o2sib, too expensive!!" << endl;
		exit(1);
	}
	//************WE MUST SPECIFY O1_FILTER****************//

	int tmp2_right=0,tmp2_wrong=0,tmp2_bad=0;
	int tmp3_right=0,tmp3_wrong=0,tmp3_bad=0;
	for(int i=0;i<sentences;i++){
		DependencyInstance* x = training_corpus->at(i);
		double* scores_o1_filter = all_scores_o1[i];
		int length = x->length();
		for(int m=1;m<length;m++){
			for(int h=0;h<length;h++){
				if(h==m)
					continue;
				//get information
				int small = GET_MIN_ONE(m,h);
				int large = GET_MAX_ONE(m,h);
				bool link_hm = (x->heads->at(m)==h);
				int noprob_hm = score_noprob(scores_o1_filter[get_index2(length,h,m)]);
				int c=-1;	//inside sibling
				if(link_hm){
				if(h>m){
					for(int ii=m+1;ii<h;ii++)
						if(x->heads->at(ii)==h){
							c = ii;
							break;
						}
				}
				else{
					for(int ii=m-1;ii>h;ii--)
						if(x->heads->at(ii)==h){
							c = ii;
							break;
						}
				}}
				//assign
				if(link_hm && c<0)
					tmp2_right++;
				else if(noprob_hm)
					tmp2_bad++;
				else
					tmp2_wrong++;
				for(int mid=small+1;mid<large;mid++){
					if(link_hm && mid==c)
						tmp3_right++;
					else if(noprob_hm || score_noprob(scores_o1_filter[get_index2(length,h,mid)]))
						tmp3_bad++;
					else
						tmp3_wrong++;
				}
			}
		}
	}
	tmpall_right=tmp2_right+tmp3_right;
	tmpall_wrong=tmp2_wrong+tmp3_wrong;
	tmpall_bad=tmp2_bad+tmp3_bad;
	printf("--Stat<all,2,3>:right(%d,%d,%d),wrong(%d,%d,%d),bad(%d,%d,%d)\n",tmpall_right,tmp2_right,tmp3_right,
			tmpall_wrong,tmp2_wrong,tmp3_wrong,tmpall_bad,tmp2_bad,tmp3_bad);

	//sweep second time and adding them
	//-allocate
	data_right = new REAL[tmpall_right*idim];
	data_wrong = new REAL[tmpall_wrong*idim];
	REAL* assign_right = data_right;
	REAL* assign_wrong = data_wrong;
	for(int i=0;i<sentences;i++){
		DependencyInstance* x = training_corpus->at(i);
		int length = x->length();
		double* scores_o1_filter = all_scores_o1[i];
		for(int m=1;m<length;m++){
			for(int h=0;h<length;h++){
				if(h==m)
					continue;
				//get information
				int small = GET_MIN_ONE(m,h);
				int large = GET_MAX_ONE(m,h);
				bool link_hm = (x->heads->at(m)==h);
				int noprob_hm = score_noprob(scores_o1_filter[get_index2(length,h,m)]);
				int c=-1;	//inside sibling
				if(link_hm){
				if(h>m){
					for(int ii=m+1;ii<h;ii++)
						if(x->heads->at(ii)==h){
							c = ii;
							break;
						}
				}
				else{
					for(int ii=m-1;ii>h;ii--)
						if(x->heads->at(ii)==h){
							c = ii;
							break;
						}
				}}
				//assign
				if(link_hm && c<0){
					feat_gen->fill_one(assign_right,x,h,m,-1);assign_right += idim;
				}
				else if(noprob_hm){}
				else{
					feat_gen->fill_one(assign_wrong,x,h,m,-1);assign_wrong += idim;
				}
				for(int mid=small+1;mid<large;mid++){
					if(link_hm && mid==c){
						feat_gen->fill_one(assign_right,x,h,m,mid);assign_right += idim;
					}
					else if(noprob_hm || score_noprob(scores_o1_filter[get_index2(length,h,mid)])){}
					else{
						feat_gen->fill_one(assign_wrong,x,h,m,mid);assign_wrong += idim;
					}
				}
			}
		}
	}

	for(int i=0;i<sentences;i++){
		delete [](all_scores_o1[i]);
	}
	delete []all_scores_o1;
	time(&now);cout << "#Finish data-gen at " << ctime(&now) << endl;
	}

	//then considering CONF_NN_resample and copy them to finish data
	if(parameters->CONF_NN_resample < 1){
		//get part of the wrong ones --- but first shuffle them
		shuffle_data(data_wrong,data_wrong,idim,idim,tmpall_wrong*idim,tmpall_wrong*idim,10);
	}
	int tmp_sumup = tmpall_wrong*parameters->CONF_NN_resample + tmpall_right;
	data = new REAL[tmp_sumup*idim];
	target = new REAL[tmp_sumup];
	memcpy(data,data_right,tmpall_right*idim*sizeof(REAL));
	memcpy(data+tmpall_right*idim,data_wrong,tmpall_wrong*parameters->CONF_NN_resample*idim*sizeof(REAL));
	for(int i=0;i<tmp_sumup;i++){
		if(i<tmpall_right)
			target[i] = 1;
		else
			target[i] = 0;
	}
	shuffle_data(data,target,idim,1,tmp_sumup*idim,tmp_sumup,10);	//final shuffle
	cout << "--Data for this iter: samples all " << tmpall_right+tmpall_wrong << " resample: " << tmp_sumup << endl;
	current = 0;
	end = tmp_sumup;
	if(!whether_o1_filter){
		delete[] data_right;
		delete[] data_wrong;
	}
}

REAL* Method7_O2sibAll::each_next_data(int* size)
{
	//size must be even number, if no universal rays, it should be that...
	if(current >= end)
		return 0;
	if(current + *size > end)
		*size = end - current;
	//!!not adding current here
	return (data+current*mach->GetIdim());
}

void Method7_O2sibAll::each_get_grad(int size)
{
	set_softmax_gradient(target+current,mach->GetDataOut(),gradient,size,mach->GetOdim());
	//!! here add it
	current += size;
}

vector<int>* Method7_O2sibAll::each_test_one(DependencyInstance* x)
{
	vector<int>* ret;
	//combine o1 scores
	if(parameters->CONF_NN_highO_o1mach.length() > 0 &&
			(parameters->CONF_NN_highO_score_combine || parameters->CONF_NN_highO_o1filter)){
		FeatureGenO1* feat_temp_o1 = new FeatureGenO1(dict,parameters->CONF_x_window,
				parameters->CONF_add_distance,parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
		double* scores_o1 = get_scores_o1(x,parameters,mach_o1,feat_temp_o1);	//same parameters
		ret = parse_o2sib(x,scores_o1);
		delete []scores_o1;
		delete feat_temp_o1;
	}
	else{
		ret = parse_o2sib(x);
	}
	return ret;
}

//maybe init embedding from o1 machine
void Method7_O2sibAll::init_embed()
{
	if(parameters->CONF_NN_highO_o1mach.length() > 0 && parameters->CONF_NN_highO_embed_init){
		//special structure
		int all = parameters->CONF_NN_we * dict->get_count();
		mach->clone_tab(mach_o1->get_tab(),all);
	}
	else
		Process::init_embed();
}
