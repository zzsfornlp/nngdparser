/*
 * Process_static.cpp
 *
 *  Created on: 2015Äê3ÔÂ18ÈÕ
 *      Author: zzs
 */

//some helpful methods
#include "Process.h"

void Process::shuffle_data(REAL* x,REAL* y,int xs,int ys,int xall,int yall,int times)
{
	//must make sure xall%xs and yall%ys == 0
	int num = xall / xs;
	int num2 = yall / ys;
	if(num!=num2 || xall%xs!=0 || yall%ys!=0)
		Error("Illegal size when shuffle");
	//shuffle 10 times
	cout << "--shuffle data " << times << " times.";
	REAL* tx = new REAL[xs];
	REAL* ty = new REAL[ys];
	int xcopy_size = xs*sizeof(REAL);
	int ycopy_size = ys*sizeof(REAL);
	for(int i=0;i<times;i++){
		for(int t=0;t<num;t++){
			int which = t+(rand()%(num-t));
			if(which==t)
				continue;
			//t->temp
			memcpy(tx,x+t*xs,xcopy_size);
			memcpy(ty,y+t*ys,ycopy_size);
			//which->t
			memcpy(x+t*xs,x+which*xs,xcopy_size);
			memcpy(y+t*ys,y+which*ys,ycopy_size);
			//temp->which
			memcpy(x+which*xs,tx,xcopy_size);
			memcpy(y+which*ys,ty,ycopy_size);
		}
	}
	delete [] tx;
	delete [] ty;
	cout << " -- Done." << endl;
}

//error function(mininize); get -1 * gradient
// -d(E)/d(zi) = (if i is target) ? 1-zi : -zi
void Process::set_softmax_gradient(const REAL* s_target,const REAL* s_output,REAL* s_gradient,int bsize,int c)
{
	const REAL *optr=s_output;
	REAL *gptr=s_gradient;
	const REAL *tptr=s_target;
	int n = bsize*c;
	REAL f1=-1.0;
	memcpy(s_gradient,s_output,n*sizeof(REAL));
	SCAL(&n,&f1,s_gradient,&inc1);
	for (int b=0; b<bsize; b++) {
		if (*tptr<0.0) ErrorN("negative index %f at %d",*tptr,b);
		int tidx=(uint) *tptr++;
		gptr[tidx] += 1.0;
		gptr+=c;
		optr+=c;
	}
}

//E=max(0,1+f(-)-f(+));  -dE/dz = 0 / 1 / -1;bsize must be even
void Process::set_pair_gradient(const REAL* s_output,REAL* s_gradient,int bsize)
{
	const REAL *optr = s_output;
	REAL *gptr = s_gradient;
	for(int b=0; b<bsize; b += 2) {
		*gptr = 0;
		*(gptr+1) = 0;
		//first right then wrong
		if((1 + *(optr+1) - (*optr)) > 0){
			*gptr = 1;
			*(gptr+1) = -1;
		}
		gptr += 2;
		optr += 2;
	}
}
