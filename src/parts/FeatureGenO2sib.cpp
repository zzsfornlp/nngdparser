/*
 * FeatureGenO2sib.cpp
 *
 *  Created on: 2015Äê3ÔÂ9ÈÕ
 *      Author: zzs
 */

#include "FeatureGenO2sib.h"

FeatureGenO2sib::FeatureGenO2sib(Dict* d,int w,int di,int apos,int d_sys):FeatureGen(d,w,di,apos,d_sys)
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


int FeatureGenO2sib::fill_one(REAL* to_fill,DependencyInstance* ins,int head,int mod,int mod_center,int no_use)
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

	//2.center
	if(mod_center<0){	//use dummy
		for(int i=mod_center-backward;i<=mod_center+backward;i++){	//**ONCE A BUG**
			if(head < mod)		*to_fill = dictionary->get_index(&dictionary->WORD_DUMMY_L,0);
			else				*to_fill = dictionary->get_index(&dictionary->WORD_DUMMY_R,0);
			to_fill ++;
		}
	}
	else{
		for(int i=mod_center-backward;i<=mod_center+backward;i++){
			if(i<0)				*to_fill = dictionary->get_index(&dictionary->WORD_START,0);	//must exist
			else if(i>=leng)	*to_fill = dictionary->get_index(&dictionary->WORD_END,0);	//must exist
			else				*to_fill = ins->index_forms->at(i);
			to_fill ++;
		}
	}
	if(pos_add){
		if(mod_center<0){	//use dummy
			for(int i=mod_center-backward;i<=mod_center+backward;i++){		//**ONCE A BUG**
				if(head < mod)	*to_fill = dictionary->get_index(&dictionary->POS_DUMMY_L,0);
				else			*to_fill = dictionary->get_index(&dictionary->POS_DUMMY_R,0);
				to_fill ++;
			}
		}
		else{
			for(int i=mod_center-backward;i<=mod_center+backward;i++){
				if(i<0)			*to_fill = dictionary->get_index(&dictionary->POS_START,0);	//must exist
				else if(i>=leng)*to_fill = dictionary->get_index(&dictionary->POS_END,0);	//must exist
				else			*to_fill = ins->index_pos->at(i);
				to_fill ++;
			}
		}
	}
	if(distance){
		if(mod_center<0){
			*to_fill = dictionary->get_index(&dictionary->DISTANCE_DUMMY,0);
			to_fill ++;
		}
		else{
			*to_fill = dictionary->get_index(head-mod_center);
			to_fill ++;
		}
	}

	//3.modifier
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

	return xdim;
}

/*************************@deprecated****************************************/
//filter based on pos
void FeatureGenO2sib::add_filter(vector<DependencyInstance*>* c)
{
	filter_map = new IntHashMap();
	int size = c->size();
	//add possible pos pairs, assuming pos index is less than 500 (depend on dictionary) and also !=0
	// -- 2*500^3 still ok for int
	// <h,m,before,2(dir)>
	for(int i=0;i<size;i++){
		DependencyInstance* x = c->at(i);
		int length = x->length();
		for(int h=0;h<length;h++){
			int before = -1;
			for(int m=h-1;m>=0;m--){
				if(x->heads->at(m) == h){
					int adding = 0;
					if(before<0){
						adding = (x->index_pos->at(h)<<(ASSUMING_MAX_POS_SHIFT+ASSUMING_MAX_POS_SHIFT+1))
										+(x->index_pos->at(m)<<(ASSUMING_MAX_POS_SHIFT+1))
										+(0<<1) + 0;
					}
					else{
						adding = (x->index_pos->at(h)<<(ASSUMING_MAX_POS_SHIFT+ASSUMING_MAX_POS_SHIFT+1))
										+(x->index_pos->at(m)<<(ASSUMING_MAX_POS_SHIFT+1))
										+(x->index_pos->at(before)<<1) + 0;
					}
					filter_map->insert(pair<int, int>(adding,0));
					before = m;
				}
			}
			before = -1;
			for(int m=h+1;m<length;m++){
				if(x->heads->at(m) == h){
					int adding = 0;
					if(before<0){
						adding = (x->index_pos->at(h)<<(ASSUMING_MAX_POS_SHIFT+ASSUMING_MAX_POS_SHIFT+1))
										+(x->index_pos->at(m)<<(ASSUMING_MAX_POS_SHIFT+1))
										+(0<<1) + 1;
					}
					else{
						adding = (x->index_pos->at(h)<<(ASSUMING_MAX_POS_SHIFT+ASSUMING_MAX_POS_SHIFT+1))
										+(x->index_pos->at(m)<<(ASSUMING_MAX_POS_SHIFT+1))
										+(x->index_pos->at(before)<<1) + 1;
					}
					filter_map->insert(pair<int, int>(adding,0));
					before = m;
				}
			}
		}
	}
}

int FeatureGenO2sib::allowed_pair(DependencyInstance* x,int head,int mod,int c)
{
	int h_i = x->index_pos->at(head);
	int m_i = x->index_pos->at(mod);
	int c_i = 0;
	if(c>=0)
		c_i = x->index_pos->at(c);
	int dir = 1;	//is direction-left-to-right
	if(mod < head)
		dir = 0;
	int test = (h_i<<(ASSUMING_MAX_POS_SHIFT+ASSUMING_MAX_POS_SHIFT+1))+(m_i<<(ASSUMING_MAX_POS_SHIFT+1))
					+(c_i<<1) + dir;
	if(filter_map->find(test) != filter_map->end())
		return 1;
	else
		return 0;
}

