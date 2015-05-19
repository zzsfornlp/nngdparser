/*
 * FeatureGebO1.cpp
 *
 *  Created on: Dec 19, 2014
 *      Author: zzs
 */

#include "FeatureGenO1.h"

FeatureGenO1::FeatureGenO1(Dict* d,int w,int di,int apos,int d_sys):FeatureGen(d,w,di,apos,d_sys)
{
	xdim = 2*w;
	if(apos)
		xdim *= 2;
	if(di){
		xdim += 1;
		if(d_sys)		//symmetric dummy node
			xdim += 1;
	}
}

int FeatureGenO1::fill_one(REAL* to_fill,DependencyInstance* ins,int head,int mod,int no_use1,int no_use2)
{
	//head-context-(distance)  |  modifier...
	int backward = window_size/2;	//window_size should be odd...
	int leng = ins->forms->size();

	//1.head
	for(int i=head-backward;i<=head+backward;i++){
		if(i<0)				*to_fill = dictionary->get_index(&dictionary->WORD_START,0);	//must exist
		else if(i>=leng)	*to_fill = dictionary->get_index(&dictionary->WORD_END,0);	//must exist
		else				*to_fill = ins->index_forms->at(i);
		to_fill ++;
	}
	if(pos_add){
		for(int i=head-backward;i<=head+backward;i++){
			if(i<0)			*to_fill = dictionary->get_index(&dictionary->POS_START,0);	//must exist
			else if(i>=leng)*to_fill = dictionary->get_index(&dictionary->POS_END,0);	//must exist
			else			*to_fill = ins->index_pos->at(i);
			to_fill ++;
		}
	}
	if(distance && distance_parent){
		*to_fill = dictionary->get_index(0);		//dummy node only for symmetric
		to_fill ++;
	}

	//2.modifier
	for(int i=mod-backward;i<=mod+backward;i++){
		if(i<0)				*to_fill = dictionary->get_index(&dictionary->WORD_START,0);	//must exist
		else if(i>=leng)	*to_fill = dictionary->get_index(&dictionary->WORD_END,0);	//must exist
		else				*to_fill = ins->index_forms->at(i);
		to_fill ++;
	}
	if(pos_add){
		for(int i=mod-backward;i<=mod+backward;i++){
			if(i<0)			*to_fill = dictionary->get_index(&dictionary->POS_START,0);	//must exist
			else if(i>=leng)*to_fill = dictionary->get_index(&dictionary->POS_END,0);	//must exist
			else			*to_fill = ins->index_pos->at(i);
			to_fill ++;
		}
	}
	if(distance){
		*to_fill = dictionary->get_index(head-mod);
		to_fill ++;
	}

	return xdim;
}

/*************************@deprecated****************************************/
void FeatureGenO1::add_filter(vector<DependencyInstance*>* c)
{
	filter_map = new IntHashMap();
	int size = c->size();
	//add possible pos pairs, assuming pos index is less than 500 and larger than 0(depend on dictionary)
	for(int i=0;i<size;i++){
		DependencyInstance* x = c->at(i);
		int length = x->length();
		for(int mod=1;mod<length;mod++){	//exclude root
			int head = x->heads->at(mod);
			int adding = x->index_pos->at(head)*ASSUMING_MAX_POS
					+ x->index_pos->at(mod);
			filter_map->insert(pair<int, int>(adding,0));
		}
	}
}

int FeatureGenO1::allowed_pair(DependencyInstance* x,int head,int mod,int c)
{
	int h_i = x->index_pos->at(head);
	int m_i = x->index_pos->at(mod);
	int test = h_i * ASSUMING_MAX_POS + m_i;
	if(filter_map->find(test) != filter_map->end())
		return 1;
	else
		return 0;
}
