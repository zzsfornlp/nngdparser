#include<iostream>
#include"DependencyInstance.h"
using namespace std;

void DependencyInstance::init(){
	forms = NULL;
	heads = NULL;
	postags = NULL;
	index_forms = NULL;
	index_pos = NULL;
	//combined_feats = NULL;
}
DependencyInstance::DependencyInstance(){
	init();
	forms = new vector<string*>();
	heads = new vector<int>();
	postags = new vector<string*>();
	//combined_feats = new vector<string*>();
}
DependencyInstance::DependencyInstance(std::vector<string*> *forms,
		std::vector<string*> *postags,std::vector<int> *heads){
	init();
	this->forms = forms;
	this->heads = heads;
	this->postags = postags;
}
int DependencyInstance::length(){
	return (int)(forms->size());
}

//printing
string DependencyInstance::toString(){
	string tmp = string();
	for(int i=0;i<length();i++){
		tmp += i;
		tmp += ":";
		tmp += *forms->at(i);
		tmp += "/";
		tmp += *postags->at(i);
		tmp += "/";
		tmp += heads->at(i);
		tmp += ";";
	}
	return tmp;
}

DependencyInstance::~DependencyInstance(){
	vector<string*>::iterator iter;
	for(iter = forms->begin(); iter != forms->end(); ++iter){
		delete (*iter);
	}
	for(iter = postags->begin(); iter != postags->end(); ++iter){
		delete (*iter);
	}
	delete(postags);
	delete(forms);
	delete(heads);
	delete(index_forms);
	delete(index_pos);
	forms = NULL;
	heads = NULL;
}

