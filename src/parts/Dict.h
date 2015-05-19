/*
 * Dict.h
 *
 *  Created on: Dec 19, 2014
 *      Author: zzs
 */

#ifndef DICT_H_
#define DICT_H_
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <string>
#include <algorithm>
#include <fstream>
#include "HashMap.h"
#include "../tools/DependencyInstance.h"
using namespace std;

#define CONS_dict_map_size 100000
#define CONS_distance_max 10

//the dictionary
class Dict{
	HashMap* maps;	//feature maps
	int dict_num;
	int real_words_start;
	//used only when building
	int remove_single;	//remove rare word -- backoff to pos -- indicate the number
	//some CONFs
	int add_distance_way;	//the method of distance dealing
	int oov_backoff;		//whether back-off oov to pos
	int all_a2z;	//whether all transform to a-z

	int distance_max;	//just const is fine
public:
	//symbols
	static string POS_START,POS_END,POS_UNK,POS_ROOTG;
	static string WORD_START,WORD_END,WORD_UNK,WORD_ROOTG;
	static string WORD_BACKOFF_POS_PREFIX;
	//for o2
	// -- this is: h-dl-c(right-arc) or h-dr-c(left-arc)
	static string WORD_DUMMY_L,WORD_DUMMY_R,POS_DUMMY_L,POS_DUMMY_R;
	static string DISTANCE_DUMMY;

	static string* tmpfunc_toa2z(string* x){
		int s = x->length();
		string* t = new string(*x);
		for(int i=0;i<s;i++){
			int one = t->at(i);
			t->at(i) = tolower(one);
		}
		return t;
	}

	string* get_distance_str(int n,int way=1);
	int get_index(string* word,string* backoff_pos);	//word or pos
	int get_index(int distance);						//distance
	int get_word_index(string* word);	//for outer use

	void write(string file);
	int get_count(){return dict_num;}
	vector<string*>* get_words_list(){	//allocate memory
		vector<string*>* tmp = new vector<string*>();
		for(HashMap::iterator i = maps->begin();i!=maps->end();i++){
			if(i->second >= real_words_start){
				tmp->push_back(i->first);
			}
		}
		return tmp;
	}

	Dict(string file);
	Dict(int remove,int distance_way,int oov_back,int allaz,int dsize=CONS_distance_max);
	~Dict(){
		delete maps;	//leak some memory, but that's alright
	}
	void construct_dictionary(vector<DependencyInstance*>*);
};


#endif /* DICT_H_ */
