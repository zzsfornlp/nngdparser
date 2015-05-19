/*
 * FeatureGenO2g.cpp
 *
 *  Created on: 2015Äê4ÔÂ21ÈÕ
 *      Author: zzs
 */

#include "FeatureGenO2g.h"

FeatureGenO2g::FeatureGenO2g(Dict* d,int w,int di,int apos,int d_sys):FeatureGen(d,w,di,apos,d_sys)
{
	xdim = 3*w;
	if(apos)
		xdim *= 2;
	if(di){
		xdim += 2;	//2 for distance
		if(d_sys)
			xdim += 1;
	}
}


int FeatureGenO2g::fill_one(REAL* to_fill,DependencyInstance* ins,int head,int mod,int g,int no_use)
{
	//head-w,mod_center-w,mod-w,head-pos,mod_center-w,mod-pos,distances
	int backward = window_size/2;	//window_size should be odd...
	int leng = ins->forms->size();

	//1.head
	for(int i=head-backward;i<=head+backward;i++){
		if(i<0)					*to_fill = dictionary->get_index(&dictionary->WORD_START,0);	//must exist
		else if(i>=leng)		*to_fill = dictionary->get_index(&dictionary->WORD_END,0);	//must exist
		else					*to_fill = ins->index_forms->at(i);
		to_fill ++;
	}
	if(pos_add){
		for(int i=head-backward;i<=head+backward;i++){
			if(i<0)				*to_fill = dictionary->get_index(&dictionary->POS_START,0);	//must exist
			else if(i>=leng)	*to_fill = dictionary->get_index(&dictionary->POS_END,0);	//must exist
			else				*to_fill = ins->index_pos->at(i);
			to_fill ++;
		}
	}
	if(distance && distance_parent){
		*to_fill = dictionary->get_index(0);		//dummy node only for symmetric
		to_fill ++;
	}

	//2.modifier
	for(int i=mod-backward;i<=mod+backward;i++){
		if(i<0)					*to_fill = dictionary->get_index(&dictionary->WORD_START,0);	//must exist
		else if(i>=leng)		*to_fill = dictionary->get_index(&dictionary->WORD_END,0);	//must exist
		else					*to_fill = ins->index_forms->at(i);
		to_fill ++;
	}
	if(pos_add){
		for(int i=mod-backward;i<=mod+backward;i++){
			if(i<0)				*to_fill = dictionary->get_index(&dictionary->POS_START,0);	//must exist
			else if(i>=leng)	*to_fill = dictionary->get_index(&dictionary->POS_END,0);	//must exist
			else				*to_fill = ins->index_pos->at(i);
			to_fill ++;
		}
	}
	if(distance){
		*to_fill = dictionary->get_index(head-mod);
		to_fill ++;
	}

	//3.grand
	if(g==0 && head==0){
		for(int i=g-backward;i<=g+backward;i++){
			*to_fill = dictionary->get_index(&dictionary->WORD_ROOTG,0);
			to_fill++;
		}
		if(pos_add){
			for(int i=g-backward;i<=g+backward;i++){
				*to_fill = dictionary->get_index(&dictionary->POS_ROOTG,0);
				to_fill++;
			}
		}
	}
	else{
	for(int i=g-backward;i<=g+backward;i++){
		if(i<0)					*to_fill = dictionary->get_index(&dictionary->WORD_START,0);	//must exist
		else if(i>=leng)		*to_fill = dictionary->get_index(&dictionary->WORD_END,0);	//must exist
		else					*to_fill = ins->index_forms->at(i);
		to_fill ++;
	}
	if(pos_add){
		for(int i=g-backward;i<=g+backward;i++){
			if(i<0)				*to_fill = dictionary->get_index(&dictionary->POS_START,0);	//must exist
			else if(i>=leng)	*to_fill = dictionary->get_index(&dictionary->POS_END,0);	//must exist
			else				*to_fill = ins->index_pos->at(i);
			to_fill ++;
		}
	}
	}
	if(distance){
		*to_fill = dictionary->get_index(g-head);
		to_fill ++;
	}

	return xdim;
}

