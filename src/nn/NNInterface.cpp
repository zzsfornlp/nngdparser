/*
 * NNInterface.cpp
 *
 *  Created on: 2015Äê3ÔÂ31ÈÕ
 *      Author: zzs
 */

#include "NNInterface.h"
#include "CslmInterface.h"
#include <fstream>

NNInterface* NNInterface::create_one(parsing_conf* p,FeatureGen* f,int outdim)
{
	if(p->CONF_NN_toolkit == string(NN_HNAME_CSLM)){
		return CslmInterface::create_one(p,f,outdim);
	}
	else
		return 0;
}

NNInterface* NNInterface::Read(string name)
{
	//read machine header
	ifstream ifs;
	ifs.open(name.c_str());
	string header;
	ifs >> header;
	ifs.close();

	if(header == string(NN_HNAME_CSLM)){
		//cslm machine
		return CslmInterface::Read(name);
	}
	else
		return 0;
}
