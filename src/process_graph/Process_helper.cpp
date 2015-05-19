/*
 * Process_helper.cpp
 *
 *  Created on: 2015Äê4ÔÂ21ÈÕ
 *      Author: zzs
 */
#include "../algorithms/Eisner.h"
#include "../algorithms/EisnerO2sib.h"
#include "../algorithms/EisnerO2g.h"
#include "../algorithms/EisnerO3g.h"
#include "../parts/Parameters.h"
#include "Process.h"

//--------------------transfrom scores only for (0,1)--------------------------
#include <cmath>
#define SET_LOG_HERE(tmp_yes,tmp_nope,ind,prob) \
	if(prob <= 0){tmp_yes[ind] = DOUBLE_LARGENEG;tmp_nope[ind] = 0;}\
	else if(prob < 1){tmp_yes[ind] = log(prob);tmp_nope[ind] = log(1-prob);}\
	else{tmp_yes[ind] = 0;tmp_nope[ind] = DOUBLE_LARGENEG;}

static void trans_o1(double* s,int len)
{
	double* tmp_yes = new double[len*len];
	double* tmp_nope = new double[len*len];
	//to log number
	for(int i=0;i<len*len;i++){
		SET_LOG_HERE(tmp_yes,tmp_nope,i,s[i]);
	}
	//sum
	for(int m=1;m<len;m++){
		double all_nope = 0;
		for(int h=0;h<len;h++){
			if(h==m)
				continue;
			all_nope += tmp_nope[get_index2(len,h,m)];
		}
		for(int h=0;h<len;h++){
			if(h==m)
				continue;
			int ind = get_index2(len,h,m);
			s[ind] = all_nope-tmp_nope[ind]+tmp_yes[ind];
		}
	}
	delete []tmp_yes;
	delete []tmp_nope;
}
static void trans_o2sib(double* s,int len)
{
	double* tmp_yes = new double[len*len*len];
	double* tmp_nope = new double[len*len*len];
	//to log number
	for(int i=0;i<len*len*len;i++){
		SET_LOG_HERE(tmp_yes,tmp_nope,i,s[i]);
	}
	//sum
	for(int m=1;m<len;m++){
		double all_nope = 0;
		for(int h=0;h<len;h++){
			if(h==m)
				continue;
			all_nope += tmp_nope[get_index2_o2sib(len,h,h,m)];
			for(int c=h+1;c<m;c++)
				all_nope += tmp_nope[get_index2_o2sib(len,h,c,m)];
			for(int c=m+1;c<h;c++)
				all_nope += tmp_nope[get_index2_o2sib(len,h,c,m)];
		}
		for(int h=0;h<len;h++){
			if(h==m)
				continue;
			int ind = get_index2_o2sib(len,h,h,m);
			s[ind] = all_nope-tmp_nope[ind]+tmp_yes[ind];
			for(int c=h+1;c<m;c++){
				int ind = get_index2_o2sib(len,h,c,m);
				s[ind] = all_nope-tmp_nope[ind]+tmp_yes[ind];
			}
			for(int c=m+1;c<h;c++){
				int ind = get_index2_o2sib(len,h,c,m);
				s[ind] = all_nope-tmp_nope[ind]+tmp_yes[ind];
			}
		}
	}
	delete []tmp_yes;
	delete []tmp_nope;
}

static void trans_o2g(double* s,int len)
{
	double* tmp_yes = new double[len*len*len];
	double* tmp_nope = new double[len*len*len];
	//to log number
	for(int i=0;i<len*len*len;i++){
		SET_LOG_HERE(tmp_yes,tmp_nope,i,s[i]);
	}
	//sum
	for(int m=1;m<len;m++){
		double all_nope = 0;
		for(int h=0;h<len;h++){
			if(h==m)
				continue;
			int small = GET_MIN_ONE(h,m);
			int large = GET_MAX_ONE(h,m);
			for(int g=0;g<small;g++)
				all_nope += tmp_nope[get_index2_o2g(len,g,h,m)];
			for(int g=large+1;g<len;g++)
				all_nope += tmp_nope[get_index2_o2g(len,g,h,m)];
			all_nope += tmp_nope[get_index2_o2g(len,0,0,m)];	//special one
		}
		for(int h=0;h<len;h++){
			if(h==m)
				continue;
			int small = GET_MIN_ONE(h,m);
			int large = GET_MAX_ONE(h,m);
			for(int g=0;g<small;g++){
				int ind = get_index2_o2g(len,g,h,m);
				s[ind] = all_nope-tmp_nope[ind]+tmp_yes[ind];
			}
			for(int g=large+1;g<len;g++){
				int ind = get_index2_o2g(len,g,h,m);
				s[ind] = all_nope-tmp_nope[ind]+tmp_yes[ind];
			}
			int ind = get_index2_o2g(len,0,0,m);
			s[ind] = all_nope-tmp_nope[ind]+tmp_yes[ind];	//special one
		}
	}
	delete []tmp_yes;
	delete []tmp_nope;
}

static void trans_o3g(double* s,int len)
{
	double* tmp_yes = new double[len*len*len*len];
	double* tmp_nope = new double[len*len*len*len];
	//to log number
	for(int i=0;i<len*len*len*len;i++){
		SET_LOG_HERE(tmp_yes,tmp_nope,i,s[i]);
	}
	//sum
	for(int m=1;m<len;m++){
		double all_nope = 0;
		for(int h=0;h<len;h++){
			if(h==m)
				continue;
			int small = GET_MIN_ONE(h,m);
			int large = GET_MAX_ONE(h,m);
			for(int g=0;g<small;g++){
				all_nope += tmp_nope[get_index2_o3g(len,g,h,h,m)];	//g,h,h,m
				for(int c=small+1;c<large;c++)
					all_nope += tmp_nope[get_index2_o3g(len,g,h,c,m)];	//g,h,c,m
			}
			for(int g=large+1;g<len;g++){
				all_nope += tmp_nope[get_index2_o3g(len,g,h,h,m)];	//g,h,h,m
				for(int c=small+1;c<large;c++)
					all_nope += tmp_nope[get_index2_o3g(len,g,h,c,m)];	//g,h,c,m
			}
			all_nope += tmp_nope[get_index2_o3g(len,0,0,0,m)];	//special one
			for(int c=m-1;c>0;c--)
				all_nope += tmp_nope[get_index2_o3g(len,0,0,c,m)];	//0,0,c,m
		}
		for(int h=0;h<len;h++){
			int ind = 0;
			if(h==m)
				continue;
			int small = GET_MIN_ONE(h,m);
			int large = GET_MAX_ONE(h,m);
			for(int g=0;g<small;g++){
				ind = get_index2_o3g(len,g,h,h,m);	//g,h,h,m
				s[ind] = all_nope-tmp_nope[ind]+tmp_yes[ind];
				for(int c=small+1;c<large;c++){
					ind = get_index2_o3g(len,g,h,c,m);	//g,h,c,m
					s[ind] = all_nope-tmp_nope[ind]+tmp_yes[ind];
				}
			}
			for(int g=large+1;g<len;g++){
				ind = get_index2_o3g(len,g,h,h,m);	//g,h,h,m
				s[ind] = all_nope-tmp_nope[ind]+tmp_yes[ind];
				for(int c=small+1;c<large;c++){
					ind = get_index2_o3g(len,g,h,c,m);	//g,h,c,m
					s[ind] = all_nope-tmp_nope[ind]+tmp_yes[ind];
				}
			}
			ind = get_index2_o3g(len,0,0,0,m);	//special one
			s[ind] = all_nope-tmp_nope[ind]+tmp_yes[ind];
			for(int c=m-1;c>0;c--){
				ind = get_index2_o3g(len,0,0,c,m);	//0,0,c,m
				s[ind] = all_nope-tmp_nope[ind]+tmp_yes[ind];
			}
		}
	}
	delete []tmp_yes;
	delete []tmp_nope;
}
