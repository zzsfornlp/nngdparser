/*
 * FeatureGenO2sib.h
 *
 *  Created on: 2015Äê3ÔÂ9ÈÕ
 *      Author: zzs
 */

#ifndef PROCESS_FEATUREGENO2SIB_H_
#define PROCESS_FEATUREGENO2SIB_H_

#include "FeatureGen.h"
class FeatureGenO2sib: public FeatureGen{
public:
	FeatureGenO2sib(Dict* d,int w,int di,int apos,int d_sys);
	virtual int fill_one(REAL*,DependencyInstance*,int head,int mod,int mod_center,int no_use=-1);
	virtual ~FeatureGenO2sib(){}
	virtual int get_order(){return 2;}

	virtual void add_filter(vector<DependencyInstance*>*);
	int allowed_pair(DependencyInstance* x,int head,int modif,int c);	//also considering directions
};



#endif /* PROCESS_FEATUREGENO2SIB_H_ */
