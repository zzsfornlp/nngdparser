/*
 * FeatureGenO3g.h
 *
 *  Created on: 2015Äê4ÔÂ20ÈÕ
 *      Author: zzs
 */

#ifndef PARTS_FEATUREGENO3G_H_
#define PARTS_FEATUREGENO3G_H_

#include "FeatureGen.h"
class FeatureGenO3g: public FeatureGen{
public:
	FeatureGenO3g(Dict* d,int w,int di,int apos,int d_sys);
	virtual int fill_one(REAL*,DependencyInstance*,int head,int mod,int mod_center,int g);
	virtual ~FeatureGenO3g(){}
	virtual int get_order(){return 2;}

	virtual void add_filter(vector<DependencyInstance*>*){}
	int allowed_pair(DependencyInstance* x,int head,int modif,int c){return -1;}
};



#endif /* PARTS_FEATUREGENO3G_H_ */
