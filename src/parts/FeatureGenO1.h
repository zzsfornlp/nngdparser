/*
 * FeatureGebO1.h
 *
 *  Created on: Dec 19, 2014
 *      Author: zzs
 */

#ifndef FEATUREGEBO1_H_
#define FEATUREGEBO1_H_
#include "FeatureGen.h"

class FeatureGenO1: public FeatureGen{
private:

public:
	FeatureGenO1(Dict* d,int w,int di,int apos,int d_sys);
	virtual int fill_one(REAL*,DependencyInstance*,int head,int mod,int no_use1=-1,int no_use2=-1);	//mod_center no use for o1
	//virtual void deal_with_corpus(vector<DependencyInstance*>*);	--- same as base-class
	virtual ~FeatureGenO1(){}

	//for extra information(1.filter )
	virtual void add_filter(vector<DependencyInstance*>*);
	virtual int allowed_pair(DependencyInstance* x,int head,int modif,int c=-1);
};

#endif /* FEATUREGEBO1_H_ */
