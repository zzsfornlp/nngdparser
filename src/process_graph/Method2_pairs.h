/*
 * Method2_pairs.h
 *
 *  Created on: 8 Jan, 2015
 *      Author: z
 */

#ifndef PARSING2_PROCESS_METHOD2_PAIRS_H_
#define PARSING2_PROCESS_METHOD2_PAIRS_H_

//method 2: the pairs(target function max(0,1+f(-)-f(+)))
// almost same as Method1
#include "Process.h"

class Method2_pairs: public Process{
protected:
	REAL* data;
	//REAL* target;		--- don't need it
	int current;
	int end;
	REAL* gradient;

protected:
	virtual int each_get_mach_outdim(){return 1;}
	virtual void each_prepare_data_oneiter();
	virtual REAL* each_next_data(int*);
	virtual void each_get_grad(int);

public:
	Method2_pairs(string conf):Process(conf){
		data = 0;
		//target = 0;
		gradient = 0;
	}
};



#endif /* PARSING2_PROCESS_METHOD2_PAIRS_H_ */
