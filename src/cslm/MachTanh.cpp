/*
 * This file is part of the continuous space language and translation model toolkit
 * for statistical machine translation and large vocabulary speech recognition.
 *
 * Copyright 2014, Holger Schwenk, LIUM, University of Le Mans, France
 *
 * The CSLM toolkit is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License version 3 as
 * published by the Free Software Foundation
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
 *
 * $Id: MachTanh.cpp,v 1.40 2014/03/25 21:52:53 schwenk Exp $
 */

using namespace std;
#include <iostream>
#include <math.h>

#include "Tools.h"
#include "MachTanh.h"
#include "Blas.h"
#ifdef CUDA
#  include "Gpu.cuh"
#endif

MachTanh::MachTanh(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw)
 : MachLin(p_idim, p_odim, p_bsize, p_nbfw, p_nbbw)
{
#ifdef BLAS_CUDA
  tmp_tanh = cuda_alloc(odim*bsize, "temporary memory for tanh machine");
#endif
}

MachTanh::MachTanh(const MachTanh &m)
 : MachLin(m)
{
#ifdef BLAS_CUDA
  tmp_tanh = cuda_alloc(odim*bsize, "temporary memory for tanh machine");
#endif
}

MachTanh::~MachTanh()
{
#ifdef BLAS_CUDA
  if (tmp_tanh) cublasFree(tmp_tanh);
#endif
}


//-----------------------------------------------
// Tools
//-----------------------------------------------

void MachTanh::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on tanh machine" << endl;
    MachLin::Info(detailed,txt);
  }
  else {
    if (drop_out>0)
      printf("%sMachTanh %d-%d, bs=%d, drop-out=%4.2f, passes=%lu/%lu", txt, idim, odim, bsize, drop_out, nb_forw, nb_backw);
    else
      printf("%sMachTanh %d-%d, bs=%d, passes=%lu/%lu", txt, idim, odim, bsize, nb_forw, nb_backw);
#ifdef BLAS_CUDA
    printf(", on GPU %d", cuda_dev);
#endif
    tm.disp(", ");
    tmh.disp(" + tanh: ");
    printf("\n");
  }
}

//-----------------------------------------------
// Training
//-----------------------------------------------

void MachTanh::Forw(int eff_bsize)
{

  if (eff_bsize<=0) eff_bsize=bsize;
  MachLin::Forw(eff_bsize);

  tmh.start();

    // apply tanh() on output
  int s=eff_bsize*odim;
#ifdef BLAS_CUDA
    // tanh = sinh/cosh = (exp x - exp -x) / (exp x + exp -x) = (exp(2*x) - 1) / (exp(2*x) + 1)
    // CUDA device already set by MachLin::Forw()
  nppsMulC_32f_I(2.0,data_out,s);		// 2*x
  nppsExp_32f_I(data_out,s);			// exp(2*x)
  nppsAddC_32f(data_out,1.0,tmp_tanh,s);	// tmp=exp(2*x)+1
  nppsSubC_32f_I(1.0,data_out,s);		// exp(2*x)-1
  nppsDiv_32f_I(tmp_tanh,data_out,s);		// (exp(2*x)-1) / (exp(2*x)+1)

    // perform drop-out
  if (drop_out>0) {
    curandGenerateUniform(cuda_gen, (float*) drop_out_rand, s);		// in (0,1]
    cuda_check_error("generating random values for drop-out");
#ifdef DEBUG
    { REAL buf[s];
    cublasGetVector(s,sizeof(REAL),drop_out_rand,1,buf,1);
    printf(" rand : %e %e .. %e %e\n", buf[0],buf[1],buf[s-2],buf[s-1]);
    }
#endif
    GpuDropOut(s, data_out, drop_out_rand, drop_out);
  }
#else
  VTANH(&s,data_out);
    // perform drop-out
  if (drop_out>0) {
    REAL coef=1.0/(1.0-drop_out);
    REAL *rptr=drop_out_rand;
    REAL *optr=data_out;
      // TODO: may be it is faster to create a mask to be multiplied with a element-wise product
    for (int i=0; i<s; i++) {
      *rptr=drand48();  // memorize random values for backw pass
      if (*rptr++<drop_out) *optr++ = 0.0;
                       else *optr++ *= coef;
    }
  }

#endif

  tmh.stop();
}

void MachTanh::Backw(const float lrate, const float wdecay, int eff_bsize)
{
    // derivate tanh activation function
    // multiply grad_hidden by derivatives of hidden layer activities (tanh)
    // grad_out = grad_out .* f'(data_out)
    //          = grad_out .* ( 1 - data_out^2 )

  if (eff_bsize<=0) eff_bsize=bsize;
  if (!grad_out)
    Error("MachTanh::Backw(): output gradient is not set");

  tmh.start();

  int d=odim*eff_bsize;
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
#endif
  VSQR(&d,data_out);
#ifdef BLAS_CUDA
# ifdef DEBUG
  { REAL buf[d];
    cublasGetVector(d,sizeof(REAL),data_out,1,buf,1);
    cublasGetVector(d,sizeof(REAL),grad_out,1,buf,1);
  }
# endif
  nppsSubCRev_32f_I(1.0,data_out,d);
  nppsMul_32f_I(data_out,grad_out,d);
# ifdef DEBUG
  { REAL buf[d];
    cublasGetVector(d,sizeof(REAL),grad_out,1,buf,1);
  }
# endif
#else
  REAL *aptr = data_out;
  REAL *gptr = grad_out;
  for (int i=0; i<d; i++) *gptr++ *= (1.0 - *aptr++);	// TODO: can we use more MKL ?
#endif

  tmh.stop();

  MachLin::Backw(lrate, wdecay, eff_bsize);
}

