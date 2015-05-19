/*
 * Process_parse_o3g.cpp
 *
 *  Created on: 2015Äê4ÔÂ23ÈÕ
 *      Author: zzs
 */

#include "Process.h"
#include "../algorithms/Eisner.h"
#include "../algorithms/EisnerO2sib.h"
#include "../algorithms/EisnerO2g.h"
#include "../algorithms/EisnerO3g.h"


double* Process::get_scores_o3g(DependencyInstance* x,parsing_conf* zp,NNInterface * zm,FeatureGen* zf,bool* whether_cut_o1)
{
	bool* score_o1 = whether_cut_o1;
	int idim = zf->get_xdim();
	int odim = zm->GetOdim();
	int whether_o1_filter = 0;
	if(whether_cut_o1 && zp->CONF_NN_highO_o1filter)
		whether_o1_filter = 1;
	// one sentence
	int length = x->forms->size();
	long num_allocated = length*length*length*length;
	long num_togo = 0;
	double *tmp_scores = new double[num_allocated];
	for(long i=0;i<length*length*length*length;i++)
		tmp_scores[i] = DOUBLE_LARGENEG;
	REAL *mach_x = new REAL[num_allocated*idim];	//num_allocated is more than needed
	REAL* assign_x = mach_x;

#define TMP_FILL_ONE(h,m,c,g){zf->fill_one(assign_x,x,h,m,c,g); assign_x += idim; num_togo += 1;}
	//1. 0,0,c,m
	for(int m=1;m<length;m++){
		//0,0,0,m
		if(!whether_o1_filter || !score_o1[get_index2(length,0,m)]){
			TMP_FILL_ONE(0,m,-1,0)
			//0,0,c,m
			for(int c=m-1;c>0;c--){
				if(!whether_o1_filter || !score_o1[get_index2(length,0,c)])
					TMP_FILL_ONE(0,m,c,0)
			}
		}
	}
	//2. g,h,c,m
	for(int h=1;h<length;h++){
		for(int m=1;m<length;m++){
			if(h==m)
				continue;
			if(!whether_o1_filter || !score_o1[get_index2(length,h,m)]){
				int small = GET_MIN_ONE(h,m);
				int large = GET_MAX_ONE(h,m);
				for(int g=0;g<small;g++){
					if(!whether_o1_filter || !score_o1[get_index2(length,g,h)]){
						TMP_FILL_ONE(h,m,-1,g)
						for(int c=small+1;c<large;c++){
							if(!whether_o1_filter || !score_o1[get_index2(length,h,c)])
								TMP_FILL_ONE(h,m,c,g)
						}
					}
				}
				for(int g=large+1;g<length;g++){
					if(!whether_o1_filter || !score_o1[get_index2(length,g,h)]){
						TMP_FILL_ONE(h,m,-1,g)
						for(int c=small+1;c<large;c++){
							if(!whether_o1_filter || !score_o1[get_index2(length,h,c)])
								TMP_FILL_ONE(h,m,c,g)
						}
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
	//1. 0,0,c,m
	for(int m=1;m<length;m++){
		//0,0,0,m
		if(!whether_o1_filter || !score_o1[get_index2(length,0,m)]){
			TMP_GET_ONE(odim,answer,assign_y)
			tmp_scores[get_index2_o3g(length,0,0,0,m)] = answer;
			//0,0,c,m
			for(int c=m-1;c>0;c--){
				if(!whether_o1_filter || !score_o1[get_index2(length,0,c)]){
					TMP_GET_ONE(odim,answer,assign_y)
					tmp_scores[get_index2_o3g(length,0,0,c,m)] = answer;
				}
			}
		}
	}
	//2. g,h,c,m
	for(int h=1;h<length;h++){
		for(int m=1;m<length;m++){
			if(h==m)
				continue;
			if(!whether_o1_filter || !score_o1[get_index2(length,h,m)]){
				int small = GET_MIN_ONE(h,m);
				int large = GET_MAX_ONE(h,m);
				for(int g=0;g<small;g++){
					if(!whether_o1_filter || !score_o1[get_index2(length,g,h)]){
						TMP_GET_ONE(odim,answer,assign_y)
						tmp_scores[get_index2_o3g(length,g,h,h,m)] = answer;
						for(int c=small+1;c<large;c++){
							if(!whether_o1_filter || !score_o1[get_index2(length,h,c)]){
								TMP_GET_ONE(odim,answer,assign_y)
								tmp_scores[get_index2_o3g(length,g,h,c,m)] = answer;
							}
						}
					}
				}
				for(int g=large+1;g<length;g++){
					if(!whether_o1_filter || !score_o1[get_index2(length,g,h)]){
						TMP_GET_ONE(odim,answer,assign_y)
						tmp_scores[get_index2_o3g(length,g,h,h,m)] = answer;
						for(int c=small+1;c<large;c++){
							if(!whether_o1_filter || !score_o1[get_index2(length,h,c)]){
								TMP_GET_ONE(odim,answer,assign_y)
								tmp_scores[get_index2_o3g(length,g,h,c,m)] = answer;
							}
						}
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
vector<int>* Process::parse_o3g(DependencyInstance* x,double* score_of_o1,double* score_of_o2sib,double* score_of_o2g)
{
	int length = x->length();
	bool *whether_cut_o1 = 0;
	if(score_of_o1 && parameters->CONF_NN_highO_o1filter){
		whether_cut_o1 = new bool[length*length];
		for(int i=0;i<length*length;i++){
			whether_cut_o1[i] = (score_noprob(score_of_o1[i])) ? true : false;
		}
	}
	//1. o3g score
	double *tmp_scores = 0;
	if(parameters->CONF_NN_highO_score_combine_o3g_self)
		tmp_scores = get_scores_o3g(x,parameters,mach,feat_gen,whether_cut_o1);
	else{
		tmp_scores = new double[length*length*length*length];
		for(int i=0;i<length*length*length*length;i++)
			tmp_scores[i] = 0;
	}
	delete []whether_cut_o1;
	if(parameters->CONF_score_prob && parameters->CONF_NN_highO_score_combine_o3g_self)
		trans_o3g(tmp_scores,length);
	//2.other scores
	if(score_of_o1 && parameters->CONF_score_prob)// && parameters->CONF_NN_highO_score_combine)	//otherwise score_of_o1 will be 0
		trans_o1(score_of_o1,length);
	if(score_of_o2sib && parameters->CONF_score_prob)// && parameters->CONF_NN_highO_score_combine_o2sib)
		trans_o2sib(score_of_o2sib,length);
	if(score_of_o2g && parameters->CONF_score_prob)// && parameters->CONF_NN_highO_score_combine_o2g)
		trans_o2g(score_of_o2g,length);

	for(int m=1;m<length;m++){
		double s_0m = 0,s_0xm=0,s_00m = 0;
		if(score_of_o1)
			s_0m = score_of_o1[get_index2(length,0,m)];
		if(score_of_o2sib)
			s_0xm = score_of_o2sib[get_index2_o2sib(length,0,0,m)];
		if(score_of_o2g)
			s_00m = score_of_o2g[get_index2_o2g(length,0,0,m)];
		tmp_scores[get_index2_o3g(length,0,0,0,m)] += s_0m + s_0xm + s_00m;
		for(int c=m-1;c>0;c--){
			if(score_of_o2sib)
				s_0xm = score_of_o2sib[get_index2_o2sib(length,0,c,m)];
			tmp_scores[get_index2_o3g(length,0,0,c,m)] += s_0m + s_0xm + s_00m;
		}
	}
	for(int s=1;s<length;s++){
		for(int t=s+1;t<length;t++){
			double s_st=0,s_ts=0;
			if(score_of_o1){
				s_st = score_of_o1[get_index2(length,s,t)];
				s_ts = score_of_o1[get_index2(length,t,s)];
			}
			for(int g=0;g<length;g++){
				if(g>=s && g<=t)	//no non-projective
					continue;
				double s_sxt=0,s_txs=0,s_gst=0,s_gts=0;
				if(score_of_o2sib){
					s_sxt = score_of_o2sib[get_index2_o2sib(length,s,s,t)];
					s_txs = score_of_o2sib[get_index2_o2sib(length,t,t,s)];
				}
				if(score_of_o2g){
					s_gst = score_of_o2g[get_index2_o2g(length,g,s,t)];
					s_gts = score_of_o2g[get_index2_o2g(length,g,t,s)];
				}
				tmp_scores[get_index2_o3g(length,g,s,s,t)] += s_st + s_sxt + s_gst;
				tmp_scores[get_index2_o3g(length,g,t,t,s)] += s_ts + s_txs + s_gts;
				for(int c=s+1;c<t;c++){
					double s_sct=0,s_tcs=0;
					if(score_of_o2sib){
						s_sct = score_of_o2sib[get_index2_o2sib(length,s,c,t)];
						s_tcs = score_of_o2sib[get_index2_o2sib(length,t,c,s)];
					}
					tmp_scores[get_index2_o3g(length,g,s,c,t)] += s_st + s_sct + s_gst;
					tmp_scores[get_index2_o3g(length,g,t,c,s)] += s_ts + s_tcs + s_gts;
				}
			}
		}
	}

	vector<int> *ret = decodeProjective_o3g(length,tmp_scores);
	delete []tmp_scores;
	return ret;
}

