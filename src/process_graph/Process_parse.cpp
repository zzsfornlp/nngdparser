/*
 * Process_parse.cpp
 *
 *  Created on: 2015Äê3ÔÂ18ÈÕ
 *      Author: zzs
 */

#include "Process.h"
#include "../algorithms/Eisner.h"
#include "../algorithms/EisnerO2sib.h"

#define TMP_GET_ONE(o_dim,a,assign) {\
	if(o_dim>1){\
		{a=0;for(int c=0;c<o_dim;c++) a+=(*assign++)*c;}\
	}\
	else{a=*assign++;}}

//-------------------- these two also static methods -----------------------------
//return Score[length][length]
double* Process::get_scores_o1(DependencyInstance* x,parsing_conf* zp,NNInterface * zm,FeatureGen* zf)
{
	int idim = zf->get_xdim();
	int odim = zm->GetOdim();
	//default order1 parsing
	int length = x->forms->size();
	double *tmp_scores = new double[length*length];
	for(int i=0;i<length*length;i++)
		tmp_scores[i] = DOUBLE_LARGENEG;
	//construct scores using nn
	int num_pair = length*(length-1);	//2 * (0+(l-1))*l/2
	int num_pair_togo = 0;
	REAL *mach_x = new REAL[num_pair*idim];
	REAL* assign_x = mach_x;
	for(int m=1;m<length;m++){
		for(int h=0;h<length;h++){
			if(m != h){
				zf->fill_one(assign_x,x,h,m);
				assign_x += idim;
				num_pair_togo ++;
			}
		}
	}
	REAL* mach_y = zm->mach_forward(mach_x,num_pair_togo);
	REAL* assign_y = mach_y;
	double answer = 0;
	for(int m=1;m<length;m++){
		for(int h=0;h<length;h++){
			if(m != h){
				TMP_GET_ONE(odim,answer,assign_y)
				tmp_scores[get_index2(length,h,m)] = answer;
			}
		}
	}
	delete []mach_x;
	delete []mach_y;
	return tmp_scores;
}

//return Score[length][length][length]
double* Process::get_scores_o2sib(DependencyInstance* x,parsing_conf* zp,NNInterface * zm,FeatureGen* zf,bool* score_o1)
{
	int idim = zf->get_xdim();
	int odim = zm->GetOdim();
	int whether_o1_filter = 0;
	if(score_o1 && zp->CONF_NN_highO_o1filter)
		whether_o1_filter = 1;
	// one sentence
	int length = x->forms->size();
	int num_allocated = length*length*length;
	int num_togo = 0;
	double *tmp_scores = new double[length*length*length];
	for(int i=0;i<length*length*length;i++)
		tmp_scores[i] = DOUBLE_LARGENEG;
	REAL *mach_x = new REAL[num_allocated*idim];	//num_allocated is more than needed
	REAL* assign_x = mach_x;
	for(int m=1;m<length;m++){
		for(int h=0;h<length;h++){
			if(h==m)
				continue;
			bool norpob_hm = score_o1[get_index2(length,h,m)];
			//h,m,-1
			if(!whether_o1_filter || !norpob_hm){
				zf->fill_one(assign_x,x,h,m,-1);
				assign_x += idim;
				num_togo += 1;
			}
			//h,m,c
			int small = GET_MIN_ONE(m,h);
			int large = GET_MAX_ONE(m,h);
			if(!whether_o1_filter || !norpob_hm){
				for(int c=small+1;c<large;c++){
					if(!whether_o1_filter || !score_o1[get_index2(length,h,c)]){
						zf->fill_one(assign_x,x,h,m,c);
						assign_x += idim;
						num_togo += 1;
					}
				}
			}
		}
	}
	//forward
	REAL* mach_y = zm->mach_forward(mach_x,num_togo);
	//and assign the scores
	REAL* assign_y = mach_y;
	double answer = 0;
	for(int m=1;m<length;m++){
		for(int h=0;h<length;h++){
			if(h==m)
				continue;
			bool norpob_hm = score_o1[get_index2(length,h,m)];
			//h,m,-1
			if(!whether_o1_filter || !norpob_hm){
				TMP_GET_ONE(odim,answer,assign_y)
				tmp_scores[get_index2_o2sib(length,h,h,m)] = answer;
			}
			//h,m,c
			int small = GET_MIN_ONE(m,h);
			int large = GET_MAX_ONE(m,h);
			if(!whether_o1_filter || !norpob_hm){
				for(int c=small+1;c<large;c++){
					if(!whether_o1_filter || !score_o1[get_index2(length,h,c)]){
						TMP_GET_ONE(odim,answer,assign_y)
						tmp_scores[get_index2_o2sib(length,h,c,m)] = answer;
					}
				}
			}
		}
	}
	delete []mach_x;
	delete []mach_y;
	return tmp_scores;
}

#include "Process_helper.cpp"	//static functions
//-------------------- parsing non-static methods -----------------------------
vector<int>* Process::parse_o1(DependencyInstance* x)
{
	double *tmp_scores = get_scores_o1(x,parameters,mach,feat_gen);
	if(parameters->CONF_score_prob)
		trans_o1(tmp_scores,x->length());
	vector<int> *ret = decodeProjective(x->length(),tmp_scores);
	delete []tmp_scores;
	return ret;
}

vector<int>* Process::parse_o2sib(DependencyInstance* x,double* score_of_o1)
{
	int length = x->length();
	bool *whether_cut_o1 = 0;
	if(score_of_o1 && parameters->CONF_NN_highO_o1filter){
		whether_cut_o1 = new bool[length*length];
		for(int i=0;i<length*length;i++){
			whether_cut_o1[i] = (score_noprob(score_of_o1[i])) ? true : false;
		}
	}
	double *tmp_scores = get_scores_o2sib(x,parameters,mach,feat_gen,whether_cut_o1);
	delete []whether_cut_o1;
	if(parameters->CONF_score_prob)
		trans_o2sib(tmp_scores,length);
	if(score_of_o1 && parameters->CONF_NN_highO_score_combine){
		if(parameters->CONF_score_prob)
			trans_o1(score_of_o1,length);
		for(int m=1;m<length;m++){
			for(int h=0;h<length;h++){
				if(m!=h){
					double score_tmp = score_of_o1[get_index2(length,h,m)];
					tmp_scores[get_index2_o2sib(length,h,h,m)] += score_tmp;
					for(int c=h+1;c<m;c++)
						tmp_scores[get_index2_o2sib(length,h,c,m)] += score_tmp;
					for(int c=m+1;c<h;c++)
						tmp_scores[get_index2_o2sib(length,h,c,m)] += score_tmp;
				}
			}
		}
	}
	vector<int> *ret = decodeProjective_o2sib(length,tmp_scores);
	delete []tmp_scores;
	return ret;
}

void Process::check_o1_filter(string m_name,string cutting)
{
	//MUST BE O1 MACH
	cout << "----- Check o1 filter(must be o1-mach)-----" << endl;
	parameters->CONF_NN_highO_o1filter_cut = atof(cutting.c_str());
	dict = new Dict(parameters->CONF_dict_file);
	mach = NNInterface::Read(m_name);
	dev_test_corpus = read_corpus(parameters->CONF_test_file);
	FeatureGenO1* feat_temp_o1 = new FeatureGenO1(dict,parameters->CONF_x_window,
			parameters->CONF_add_distance,parameters->CONF_add_pos,parameters->CONF_add_distance_parent);
	feat_temp_o1->deal_with_corpus(dev_test_corpus);
	if(mach->GetIdim() != feat_temp_o1->get_xdim()){
		cout << "Wrong mach...\n";
		exit(1);
	}
	int token_num = 0;	//token number
	int filter_wrong_count = 0;
	for(int ii=0;ii<dev_test_corpus->size();ii++){
		if(ii%100 == 0)
			cout << filter_wrong_count << "/" << token_num << endl;
		DependencyInstance* x = dev_test_corpus->at(ii);
		int length = x->forms->size();
		token_num += length - 1;
		double* scores_o1 = get_scores_o1(x,parameters,mach,feat_temp_o1);	//same parameters
		for(int i=1;i<length;i++){
			if(score_noprob(scores_o1[get_index2(length,x->heads->at(i),i)]))
				filter_wrong_count++;
			token_num++;
		}
		delete []scores_o1;
	}
	cout << "FINAL:" << filter_wrong_count << "/" << token_num << endl;
}


