/*
 * FeatureGen.cpp
 *
 *  Created on: 2015Äê3ÔÂ9ÈÕ
 *      Author: zzs
 */

#include "FeatureGen.h"

void FeatureGen::deal_with_corpus(vector<DependencyInstance*>* c)
{
	if(c->at(0)->index_forms)
		return;
	int size = c->size();
	for(int i=0;i<size;i++){
		DependencyInstance* x = c->at(i);
		int length = x->length();
		x->index_forms = new vector<int>();
		x->index_pos = new vector<int>();
		for(int ii=0;ii<length;ii++){
			(x->index_forms)->push_back(dictionary->get_index(x->forms->at(ii),x->postags->at(ii)));
			(x->index_pos)->push_back(dictionary->get_index(x->postags->at(ii),0));
		}
	}
}

//io
void FeatureGen::write_extra_info(string file)
{
	//warning when error
	printf("-Writing feat-file to %s,size %d.\n",file.c_str(),filter_map->size());
	ofstream fout;
	fout.open(file.c_str(),ofstream::out);
	//fout << filter_map->size() << "\n";
	for(IntHashMap::iterator i = filter_map->begin();i!=filter_map->end();i++){
		fout << i->first << endl;
	}
	fout.close();
	printf("-Writing finished.\n");
}

void FeatureGen::read_extra_info(string file)
{
	filter_map = new IntHashMap();
	printf("-Reading feat-file from %s.\n",file.c_str());
	ifstream fin;
	fin.open(file.c_str(),ifstream::in);
	while(fin){
		int temp;
		fin >> temp;
		filter_map->insert(pair<int, int>(temp,0));
	}
	printf("-Reading finished,size %d.\n",filter_map->size());
}
