/*
 * Dict.cpp
 *
 *  Created on: Dec 19, 2014
 *      Author: zzs
 */

#include "Dict.h"
#include "Parameters.h"

string Dict::POS_START = "<pos-s>";
string Dict::POS_END = "<pos-e>";
string Dict::POS_UNK = "<pos-unk>";
string Dict::POS_ROOTG = "<pos-rootg>";	//for high-order grandnode of root
string Dict::WORD_START = "<w-s>";
string Dict::WORD_END = "<w-e>";
string Dict::WORD_UNK = "<w-unk>";
string Dict::WORD_ROOTG = "<w-rootg>";
string Dict::WORD_BACKOFF_POS_PREFIX = "_sth_";

string Dict::WORD_DUMMY_L = "<w-dl>";
string Dict::WORD_DUMMY_R = "<w-dr>";
string Dict::POS_DUMMY_L = "<pos-dl>";
string Dict::POS_DUMMY_R = "<pos-dr>";
string Dict::DISTANCE_DUMMY = "_distance_dummy_";

Dict::Dict(int remove,int distance_way,int oov_back,int allaz,int dsize){
	maps = new HashMap(CONS_dict_map_size);
	all_a2z = allaz;
	remove_single = remove;
	distance_max = dsize;
	add_distance_way = distance_way;
	oov_backoff = oov_back;
	dict_num = 0;
	real_words_start = 0;
}

string* Dict::get_distance_str(int n,int way)
{
	char temp[100];
	string prefix = "_distance_";
	if(n>=distance_max)
		n = distance_max;
	else if(n<=-1*distance_max)
		n = -1*distance_max;

	//really really bad design...
	switch(way){
		case 1: break;
		case 2: //same as the Mcd... ,5,10
			if(n<distance_max && n>5)
				n=5;
			else if(n>-1*distance_max && n<-5)
				n=-5;
			break;
		default:
			break;
	}
	sprintf(temp,"_distance_%d",n);
	return new string(temp);
}

int Dict::get_index(int d)
{
	string* temp = get_distance_str(d,add_distance_way);
	int x = maps->find(temp)->second;	//must exist
	delete temp;
	return x;
}

int Dict::get_index(string* word,string* backoff_pos)
{
	//2 ways:(backoff_pos=0 for pos, else for word)
	if(all_a2z)
		word = tmpfunc_toa2z(word);
	HashMap::iterator iter = maps->find(word);
	if(all_a2z)
		delete word;

	if(iter == maps->end()){
		if(backoff_pos == 0){
			//for purely pos
			return maps->find(&POS_UNK)->second;
		}
		else if(oov_backoff){
			int i;
			string* temp = new string(WORD_BACKOFF_POS_PREFIX+*backoff_pos);
			HashMap::iterator iter2 = maps->find(temp);
			if(iter2 == maps->end())
				i = maps->find(&WORD_UNK)->second;
			else
				i = iter2->second;
			delete temp;
			return i;
		}
		else
			return maps->find(&WORD_UNK)->second;
	}
	else
		return iter->second;
}

int Dict::get_word_index(string* word)
{
	int ret = -1;
	if(all_a2z)
		word = tmpfunc_toa2z(word);
	HashMap::iterator iter = maps->find(word);
	if(iter == maps->end()){}
	else
		ret = iter->second;
	if(all_a2z)
		delete word;
	return ret;
}

void Dict::construct_dictionary(vector<DependencyInstance*>* corpus){
	printf("-Start to build dictionary\n");
	// !!When constructing the dictionary, includes all the features (no harm) ...
	int num_distance=0,num_pos=0,num_words=0;
	//1.first add distance words
	maps->insert(pair<string*, int>(&DISTANCE_DUMMY,dict_num++));
	num_distance++;
	for(int i=-1*distance_max;i<=distance_max;i++){
		string * dis = get_distance_str(i);
		maps->insert(pair<string*, int>(dis,dict_num++));
		num_distance++;
	}

	//2.add pos
	vector<string*> real_pos;
	maps->insert(pair<string*, int>(&POS_START,dict_num++));
	maps->insert(pair<string*, int>(&POS_END,dict_num++));
	maps->insert(pair<string*, int>(&POS_UNK,dict_num++));
	maps->insert(pair<string*, int>(&POS_ROOTG,dict_num++));
	maps->insert(pair<string*, int>(&POS_DUMMY_L,dict_num++));
	maps->insert(pair<string*, int>(&POS_DUMMY_R,dict_num++));
	num_pos += 6;
	int corpus_size = corpus->size();
	for(int i=0;i<corpus_size;i++){
		DependencyInstance* one = corpus->at(i);
		vector<string*>* one_pos = one->postags;
		int sen_length = one_pos->size();
		for(int j=0;j<sen_length;j++){
			string* to_find = one_pos->at(j);
			HashMap::iterator iter = maps->find(to_find);
			if(iter == maps->end()){
				maps->insert(pair<string*, int>(to_find,dict_num++));
				num_pos++;
				real_pos.push_back(to_find);
			}
		}
	}

	//3.add words
	//3.1-pseudo
	maps->insert(pair<string*, int>(&WORD_START,dict_num++));
	maps->insert(pair<string*, int>(&WORD_END,dict_num++));
	maps->insert(pair<string*, int>(&WORD_UNK,dict_num++));
	maps->insert(pair<string*, int>(&WORD_ROOTG,dict_num++));
	maps->insert(pair<string*, int>(&WORD_DUMMY_L,dict_num++));
	maps->insert(pair<string*, int>(&WORD_DUMMY_R,dict_num++));
	num_words += 6;
	//3.2-backoff_pos
	for(vector<string*>::iterator i=real_pos.begin();i!=real_pos.end();i++){
		string* new_w = new string(WORD_BACKOFF_POS_PREFIX+(**i));
		num_words++;
		maps->insert(pair<string*, int>(new_w,dict_num++));
	}
	real_words_start = dict_num;	//now the rest is real words

	//3.3-real words
	HashMap real_words_map(CONS_dict_map_size);
	vector<string*> real_words;
	vector<int> real_words_count;
	int real_word_num = 0;
	for(int i=0;i<corpus_size;i++){
		DependencyInstance* one = corpus->at(i);
		vector<string*>* one_form = one->forms;
		int sen_length = one_form->size();
		for(int j=0;j<sen_length;j++){
			string* to_find = one_form->at(j);
			if(all_a2z){
				to_find = tmpfunc_toa2z(to_find);
			}
			HashMap::iterator iter = real_words_map.find(to_find);
			if(iter == real_words_map.end()){
				real_words_map.insert(pair<string*, int>(to_find,real_word_num++));
				num_words++;
				real_words.push_back(to_find);
				real_words_count.push_back(1);
			}
			else{
				int temp_n = iter->second;
				real_words_count[temp_n] ++;	//add counting
			}
		}
	}
	real_words_map.clear();

	//4.deal with statistics and removing
	printf("-Finish dictionary building, all is %d,distance %d,pos %d,words %d.\n",
			dict_num+real_word_num,num_distance,num_pos,num_words);
	if(1){
		//counting frequencies
		printf("--Trying to get more specific statistics of words-count:\n");
		vector<int> temp_words_count = real_words_count;
		std::sort(temp_words_count.begin(),temp_words_count.end());	//sorting
		int current_freq = 0,real_word_size = temp_words_count.size(),current_count=0;
		for(int i=0;i<real_word_size;i++){
			if(temp_words_count[i] != current_freq){
				printf("[%d,%d] ",current_freq,current_count);
				current_count = 1;
				current_freq = temp_words_count[i];
			}
			else
				current_count++;
		}
		printf("[%d,%d]\n",current_freq,current_count);

		int to_remove = 0;
		if(remove_single){
			printf("--Trying to remove single count words:\n");
			for(int i=0;i<real_words.size();i++){
				if(real_words_count[i] <= remove_single){
					to_remove++;
				}
				else{
					maps->insert(pair<string*, int>(real_words[i],dict_num++));
				}
			}
			printf("-Remove single count words %d:\n",to_remove);
			printf("--Final finish dictionary building, all is %d,distance %d,pos %d,words %d.\n",
						dict_num,num_distance,num_pos,num_words-to_remove);
			return;	//here get out early
		}
	}

	//final adding
	for(int i=0;i<real_words.size();i++){
		maps->insert(pair<string*, int>(real_words[i],dict_num++));
	}
	printf("--Final finish dictionary building, all is %d,distance %d,pos %d,words %d.\n",
				dict_num,num_distance,num_pos,num_words);
}

//io
void Dict::write(string file)
{
	//warning when error
	printf("-Writing dict to %s.\n",file.c_str());
	ofstream fout;
	fout.open(file.c_str(),ofstream::out);
	fout << dict_num << " " << real_words_start << " " << distance_max << " ";
	fout << add_distance_way << " " << oov_backoff << " " << all_a2z << "\n";
	string** all = new string*[dict_num];
	for(HashMap::iterator i = maps->begin();i!=maps->end();i++){
		if(i->second >= dict_num){
			//error
			fprintf(stderr,"Error with dictionary size...\n");
		}
		all[i->second] = i->first;
	}
	for(int i=0;i<dict_num;i++)
		fout << *(all[i]) << "\n";
	fout.close();
	delete []all;
	printf("-Writing finished.\n");
}

Dict::Dict(string file)
{
	maps = new HashMap(CONS_dict_map_size);
	//here no need for real-word-list; only for init-build
	printf("-Reading dict from %s.\n",file.c_str());
	ifstream fin;
	fin.open(file.c_str(),ifstream::in);
	fin >> dict_num >> real_words_start >> distance_max >> add_distance_way >> oov_backoff >> all_a2z;
	for(int i=0;i<dict_num;i++){
		string t;
		fin >> t;
		maps->insert(pair<string*, int>(new string(t),i));
	}
	printf("-Reading dict finished.\n");
}
