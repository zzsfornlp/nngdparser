/*
 * Process_parse_g.cpp
 *
 *  Created on: 2015Äê4ÔÂ21ÈÕ
 *      Author: zzs
 */

#include "Process.h"
#include "../algorithms/Eisner.h"
#include "../algorithms/EisnerO2g.h"

//return Score[length][length][length]
double* Process::get_scores_o2g(DependencyInstance* x,parsing_conf* zp,NNInterface * zm,FeatureGen* zf,bool* whether_cut_o1)
{
	int idim = zf->get_xdim();
	int odim = zm->GetOdim();
	int whether_o1_filter = 0;
	if(whether_cut_o1 && zp->CONF_NN_highO_o1filter)
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
		//1.special 0,0,m
		if(!whether_o1_filter || !whether_cut_o1[get_index2(length,0,m)]){
			zf->fill_one(assign_x,x,0,m,0);
			assign_x += idim;
			num_togo += 1;
		}
		//2.others
		for(int h=0;h<length;h++){
			if(h==m)
				continue;
			if(!whether_o1_filter || !whether_cut_o1[get_index2(length,h,m)]){
				int small = GET_MIN_ONE(m,h);
				int large = GET_MAX_ONE(m,h);
				for(int g=0;g<length;g++){
					if(g>=small && g<=large)
						continue;
					if(!whether_o1_filter || !whether_cut_o1[get_index2(length,g,h)]){
						zf->fill_one(assign_x,x,h,m,g);
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
#define TMP_GET_ONE(o_dim,a,assign) {\
	if(o_dim>1){\
		{a=0;for(int c=0;c<o_dim;c++) a+=(*assign++)*c;}\
	}\
	else{a=*assign++;}}

	REAL* assign_y = mach_y;
	double answer = 0;
	for(int m=1;m<length;m++){
		//1.special 0,0,m
		if(!whether_o1_filter || !whether_cut_o1[get_index2(length,0,m)]){
			TMP_GET_ONE(odim,answer,assign_y)
			tmp_scores[get_index2_o2g(length,0,0,m)] = answer;
		}
		//2.others
		for(int h=0;h<length;h++){
			if(h==m)
				continue;
			if(!whether_o1_filter || !whether_cut_o1[get_index2(length,h,m)]){
				int small = GET_MIN_ONE(m,h);
				int large = GET_MAX_ONE(m,h);
				for(int g=0;g<length;g++){
					if(g>=small && g<=large)
						continue;
					if(!whether_o1_filter || !whether_cut_o1[get_index2(length,g,h)]){
						TMP_GET_ONE(odim,answer,assign_y)
						tmp_scores[get_index2_o2g(length,g,h,m)] = answer;
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
vector<int>* Process::parse_o2g(DependencyInstance* x,double* score_of_o1)
{
	int length = x->length();
	bool *whether_cut_o1 = 0;
	if(score_of_o1 && parameters->CONF_NN_highO_o1filter){
		whether_cut_o1 = new bool[length*length];
		for(int i=0;i<length*length;i++){
			whether_cut_o1[i] = (score_noprob(score_of_o1[i])) ? true : false;
		}
	}
	double *tmp_scores = get_scores_o2g(x,parameters,mach,feat_gen,whether_cut_o1);
	delete []whether_cut_o1;
	if(parameters->CONF_score_prob)
		trans_o2g(tmp_scores,length);
	if(score_of_o1 && parameters->CONF_NN_highO_score_combine){
		if(parameters->CONF_score_prob)
			trans_o1(score_of_o1,length);
		for(int m=1;m<length;m++){
			for(int h=0;h<length;h++){
				if(h!=m){
					double score_tmp = score_of_o1[get_index2(length,h,m)];
					int small = GET_MIN_ONE(h,m);
					int large = GET_MAX_ONE(h,m);
					for(int g=0;g<small;g++)
						tmp_scores[get_index2_o2g(length,g,h,m)] += score_tmp;
					for(int g=large+1;g<length;g++)
						tmp_scores[get_index2_o2g(length,g,h,m)] += score_tmp;
				}
			}
			double score_tmp = score_of_o1[get_index2(length,0,m)];
			tmp_scores[get_index2_o2g(length,0,0,m)] += score_tmp;
		}
	}
	vector<int> *ret = decodeProjective_o2g(length,tmp_scores);
	delete []tmp_scores;
	return ret;
}
