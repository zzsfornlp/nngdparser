/*
 * NNInterface.h
 *
 *  Created on: 2015Äê3ÔÂ31ÈÕ
 *      Author: zzs
 */

#ifndef NN_NNINTERFACE_H_
#define NN_NNINTERFACE_H_

#include <vector>
#include <stdexcept>
using namespace std;
//machines
#define NN_HNAME_CSLM "HPerf"		//1

#include "../cslm/Tools.h"
#include "../cslm/Mach.h"
#include "../parts/Parameters.h"
#include "../parts/FeatureGen.h"

#define NNERROR_NotImplemented "NNERROR_NotImplemented"
#define NNERROR_WiderThanWidth "NNERROR_WiderThanWidth"
#define NNERROR_InternalError "NNERROR_InternalError"

#define ACT_TANH 0
#define ACT_HTANH 1

class NNInterface{
protected:
	void nnError(string what)
	{
		cerr << what << endl;
		throw runtime_error(what);
	}
public:
	//data
	virtual void SetDataIn(REAL *data)=0;
	virtual REAL* GetDataOut()=0;
	virtual int GetIdim()=0;
	virtual int GetOdim()=0;
	virtual int GetWidth()=0;	//the current number of instances
	virtual void SetWidth()=0;	//only for those can change
	//forw-backw
	virtual void Forw(int)=0;	//forward bsize
	virtual void Backw(const float lrate, const float wdecay, int bs)=0;
	virtual void SetGradOut(REAL *data)=0;
	virtual ulong GetNbBackw()=0;
	//specail-backws
	virtual void Backw_store(const float lrate, const float wdecay, int bs)=0;	//don't update --- batch mode
	virtual void Backw_update()=0;
	//test-time --> should perform fast calculating using pre-calculated info if possible
	virtual REAL* mach_forward(REAL* assign,int all)=0;	//allocated here
	//tabs
	virtual REAL* get_tab()=0;
	virtual void set_tab(REAL* x)=0;
	virtual void clone_tab(REAL* x,int all)=0;
	//precalc
	virtual void pre_calc()=0;
	virtual void DEBUG_pre_calc()=0;
	//others
	virtual void Write(string name)=0;
	static NNInterface* Read(string name);
	static NNInterface* create_one(parsing_conf* p,FeatureGen* f,int outdim);
	virtual ~NNInterface(){}
};

#endif /* NN_NNINTERFACE_H_ */
