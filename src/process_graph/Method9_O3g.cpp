/*
 * Method9_O3g.cpp
 *
 *  Created on: 2015Äê4ÔÂ23ÈÕ
 *      Author: zzs
 */

#include "Method9_O3g.h"
#include "../algorithms/Eisner.h"
#include <cstdio>
#include <cstdlib>

void Method9_O3g::each_prepare_data_oneiter()
{
	delete []data;
	delete []target;
	delete []gradient;
	//for gradient
	gradient = new REAL[mach->GetWidth()*mach->GetOdim()];
	mach->SetGradOut(gradient);
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

	//************WE MUST SPECIFY O1_FILTER****************//
	if(!whether_o1_filter){
		cout << "No o1-filter for o2g, too expensive!!" << endl;
		exit(1);
	}
	//************WE MUST SPECIFY O1_FILTER****************//

	if(data_right==0){
	//1.o1-filter (MUST HAVE)
	FeatureGenO1* feat_temp_o1 = new FeatureGenO1(dict,parameters->CONF_x_window,
					parameters->CONF_add_distance,parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
	double** all_scores_o1 = new double*[sentences];
	int all_tokens_train=0,all_token_filter_wrong=0;
	for(int i=0;i<sentences;i++){
			all_scores_o1[i] = 0;
			DependencyInstance* x = training_corpus->at(i);
			all_scores_o1[i] = get_scores_o1(x,parameters,mach_o1,feat_temp_o1);
			double* scores_o1_filter = all_scores_o1[i];
			all_tokens_train += x->length();
			for(int i2=1;i2<x->length();i2++){	//ignore root
				if(score_noprob(scores_o1_filter[get_index2(x->length(),x->heads->at(i2),i2)]))
					all_token_filter_wrong ++;
		}
	}
	cout << "For o1 filter: all " << all_tokens_train << ";filter wrong " << all_token_filter_wrong << endl;
	time_t now;
	time(&now);cout << "#Finish o1-filter at " << ctime(&now) << flush;

	//2.first pass --- figure out the numbers
	int tmp2_right=0,tmp2_wrong=0,tmp2_bad=0;
	int tmp3_right=0,tmp3_wrong=0,tmp3_bad=0;
	for(int i=0;i<sentences;i++){
		DependencyInstance* x = training_corpus->at(i);
		double* scores_o1_filter = all_scores_o1[i];
		int length = x->length();
		for(int m=1;m<length;m++){
			//2.1 special (0,0,c,m)	when h==0
			int noprob_0m = score_noprob(scores_o1_filter[get_index2(length,0,m)]);
			int link_0m = (x->heads->at(m)==0);
			int c = -1;
			for(int mid=m-1;mid>0;mid--){
				if(x->heads->at(mid)==0){
					c = mid;
					break;
				}
			}
			if(link_0m && c<0)
				tmp2_right++;
			else if(noprob_0m)
				tmp2_bad++;
			else
				tmp2_wrong++;
			for(int mid=1;mid<m;mid++){
				if(link_0m && mid==c)
					tmp3_right++;
				else if(noprob_0m || score_noprob(scores_o1_filter[get_index2(length,0,mid)]))
					tmp3_bad++;
				else
					tmp3_wrong++;
			}
			//2.2. ordinary ones
			for(int h=1;h<length;h++){	//h>=1
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
				//for g and c
				for(int g=0;g<length;g++){
					if(g==h || g==m || g==c)
						continue;
					bool link_gh = (x->heads->at(h)==g);
					int noprob_gh = score_noprob(scores_o1_filter[get_index2(length,g,h)]);
					int nonproj_g = (g>=small && g<=large);
					if(link_hm && link_gh && c<0)
						tmp2_right++;
					else if(noprob_hm || noprob_gh || nonproj_g)
						tmp2_bad++;
					else
						tmp2_wrong++;
					for(int mid=small+1;mid<large;mid++){
						if(link_hm && link_gh && mid==c)
							tmp3_right++;
						else if(noprob_hm || noprob_gh || nonproj_g || score_noprob(scores_o1_filter[get_index2(length,h,mid)]))
							tmp3_bad++;
						else
							tmp3_wrong++;
					}
				}
			}
		}
	}
	tmpall_right=tmp2_right+tmp3_right;
	tmpall_wrong=tmp2_wrong+tmp3_wrong;
	tmpall_bad=tmp2_bad+tmp3_bad;
	printf("--Stat<all,2,3>:right(%d,%d,%d),wrong(%d,%d,%d),bad(%d,%d,%d)\n",tmpall_right,tmp2_right,tmp3_right,
			tmpall_wrong,tmp2_wrong,tmp3_wrong,tmpall_bad,tmp2_bad,tmp3_bad);

	//3.sweep second time and adding them
	//-allocate
	data_right = new REAL[tmpall_right*idim];
	data_wrong = new REAL[tmpall_wrong*idim];
	REAL* assign_right = data_right;
	REAL* assign_wrong = data_wrong;
	for(int i=0;i<sentences;i++){
		DependencyInstance* x = training_corpus->at(i);
		double* scores_o1_filter = all_scores_o1[i];
		int length = x->length();
		for(int m=1;m<length;m++){
			//2.1 special (0,0,c,m)	when h==0
			int noprob_0m = score_noprob(scores_o1_filter[get_index2(length,0,m)]);
			int link_0m = (x->heads->at(m)==0);
			int c = -1;
			for(int mid=m-1;mid>0;mid--){
				if(x->heads->at(mid)==0){
					c = mid;
					break;
				}
			}
			if(link_0m && c<0){
				feat_gen->fill_one(assign_right,x,0,m,-1,0);assign_right += idim;
			}
			else if(noprob_0m){}
			else{
				feat_gen->fill_one(assign_wrong,x,0,m,-1,0);assign_wrong += idim;
			}
			for(int mid=1;mid<m;mid++){
				if(link_0m && mid==c){
					feat_gen->fill_one(assign_right,x,0,m,mid,0);assign_right += idim;
				}
				else if(noprob_0m || score_noprob(scores_o1_filter[get_index2(length,0,mid)])){}
				else{
					feat_gen->fill_one(assign_wrong,x,0,m,mid,0);assign_wrong += idim;
				}
			}
			//2.2. ordinary ones
			for(int h=1;h<length;h++){	//h>=1
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
				//for g and c
				for(int g=0;g<length;g++){
					if(g==h || g==m || g==c)
						continue;
					bool link_gh = (x->heads->at(h)==g);
					int noprob_gh = score_noprob(scores_o1_filter[get_index2(length,g,h)]);
					int nonproj_g = (g>=small && g<=large);
					if(link_hm && link_gh && c<0){
						feat_gen->fill_one(assign_right,x,h,m,-1,g);assign_right += idim;
					}
					else if(noprob_hm || noprob_gh || nonproj_g){}
					else{
						feat_gen->fill_one(assign_wrong,x,h,m,-1,g);assign_wrong += idim;
					}
					for(int mid=small+1;mid<large;mid++){
						if(link_hm && link_gh && mid==c){
							feat_gen->fill_one(assign_right,x,h,m,mid,g);assign_right += idim;
						}
						else if(noprob_hm || noprob_gh || nonproj_g || score_noprob(scores_o1_filter[get_index2(length,h,mid)])){}
						else{
							feat_gen->fill_one(assign_wrong,x,h,m,mid,g);assign_wrong += idim;
						}
					}
				}
			}
		}
	}
	for(int i=0;i<sentences;i++){
		delete [](all_scores_o1[i]);
	}
	delete []all_scores_o1;
	time(&now);cout << "#Finish data-gen at " << ctime(&now) << flush;
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
	cout << "--M9, Data for this iter: samples all " << tmpall_right+tmpall_wrong << " resample: " << tmp_sumup << endl;
	current = 0;
	end = tmp_sumup;
}

REAL* Method9_O3g::each_next_data(int* size)
{
	//size must be even number, if no universal rays, it should be that...
	if(current >= end)
		return 0;
	if(current + *size > end)
		*size = end - current;
	//!!not adding current here
	return (data+current*mach->GetIdim());
}

void Method9_O3g::each_get_grad(int size)
{
	set_softmax_gradient(target+current,mach->GetDataOut(),gradient,size,mach->GetOdim());
	//!! here add it
	current += size;
}

vector<int>* Method9_O3g::each_test_one(DependencyInstance* x)
{
	vector<int>* ret;
	int length = x->length();
	double *scores_o1 = 0;
	double *scores_o2sib = 0;
	double *scores_o2g = 0;
	bool *whether_cut_o1 = 0;
	//combine scores
	if(parameters->CONF_NN_highO_o1mach.length() > 0 &&
			(parameters->CONF_NN_highO_score_combine || parameters->CONF_NN_highO_o1filter)){
		FeatureGenO1* feat_temp_o1 = new FeatureGenO1(dict,parameters->CONF_x_window,
				parameters->CONF_add_distance,parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
		scores_o1 = get_scores_o1(x,parameters,mach_o1,feat_temp_o1);	//same parameters
		whether_cut_o1 = new bool[length*length];
		for(int i=0;i<length*length;i++){
			whether_cut_o1[i] = (score_noprob(scores_o1[i])) ? true : false;
		}
		delete feat_temp_o1;
	}
	if(parameters->CONF_NN_highO_o2sibmach.length() > 0 && parameters->CONF_NN_highO_score_combine_o2sib){
		FeatureGenO2sib* feat_temp_o2sib = new FeatureGenO2sib(dict,parameters->CONF_x_window,
				parameters->CONF_add_distance,parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
		scores_o2sib = get_scores_o2sib(x,parameters,mach_o2sib,feat_temp_o2sib,whether_cut_o1);	//same parameters
		delete feat_temp_o2sib;
	}
	if(parameters->CONF_NN_highO_o2gmach.length() > 0 && parameters->CONF_NN_highO_score_combine_o2g){
		FeatureGenO2g* feat_temp_o2g = new FeatureGenO2g(dict,parameters->CONF_x_window,
				parameters->CONF_add_distance,parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
		scores_o2g = get_scores_o2g(x,parameters,mach_o2g,feat_temp_o2g,whether_cut_o1);	//same parameters
		delete feat_temp_o2g;
	}
	ret = parse_o3g(x,scores_o1,scores_o2sib,scores_o2g);
	delete []scores_o1;
	delete []scores_o2sib;
	delete []scores_o2g;
	delete []whether_cut_o1;
	return ret;
}
