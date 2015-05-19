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
 * $Id: MachSoftmaxStable.cpp,v 1.11 2014/03/25 21:52:53 schwenk Exp $
 */

using namespace std;
#include <iostream>
#include <math.h>

#include "Tools.h"
#include "MachSoftmaxStable.h"
#include "Blas.h"
#ifdef BLAS_CUDA
# include "Gpu.cuh"
#endif


MachSoftmaxStable::MachSoftmaxStable(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw)
 : MachLin(p_idim, p_odim, p_bsize, p_nbfw, p_nbbw)
{
#if defined(BLAS_CUDA) && defined(BLAS_CUDA_NPPS_SUM)
  int nbytes=0;
  cudaSetDevice(cuda_dev);
  nppsSumGetBufferSize_32f(odim, &nbytes);
  gpu_sum_buf = nppsMalloc_8u(nbytes);
#endif
#ifdef BLAS_CUDA
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props, cuda_dev);
  if(props.warpSize != 32){
    Error("KernelSoftmax used by MachSoftmaxStable suppose a wrapSize of 32. The code will return wrong result if run!");
  } 
#endif
}

MachSoftmaxStable::MachSoftmaxStable(const MachSoftmaxStable &m)
 : MachLin(m)
{
#if defined(BLAS_CUDA) && defined(BLAS_CUDA_NPPS_SUM)
  int nbytes=0;
  nppsSumGetBufferSize_32f(odim, &nbytes);
  gpu_sum_buf = nppsMalloc_8u(nbytes);
#endif
#ifdef BLAS_CUDA
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props, cuda_dev);
  if(props.warpSize != 32){
    Error("KernelSoftmax used by MachSoftmaxStable suppose a wrapSize of 32. The code will return wrong result if run!");
  }
#endif
}

MachSoftmaxStable::~MachSoftmaxStable()
{
#if defined(BLAS_CUDA) && defined(BLAS_CUDA_NPPS_SUM)
  cudaSetDevice(cuda_dev);
  if (gpu_sum_buf) nppsFree(gpu_sum_buf);
#endif
}

//-----------------------------------------------
// Tools
//-----------------------------------------------

void MachSoftmaxStable::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on softmax stable machine" << endl;
    MachLin::Info(detailed);
  }
  else {
    printf("%sMachSoftmaxStable %d-%d, bs=%d, passes=%lu/%lu", txt,idim, odim, bsize, nb_forw, nb_backw);
#ifdef BLAS_CUDA
    printf(", on GPU %d", cuda_dev);
#endif
    tm.disp(", ");
    tmn.disp(" + norm: ");
    printf("\n");
  }
}

//-----------------------------------------------
// Training
//-----------------------------------------------

void MachSoftmaxStable::Forw(int eff_bsize)
{

  if (eff_bsize<=0) eff_bsize=bsize;
  MachLin::Forw(eff_bsize);

  tmn.start();

    // softmax stable normalization
#ifdef BLAS_CUDA
    // device already set by MachLin::Forw()
  GpuMachSoftmaxStableForw(eff_bsize,odim,data_out);
#else
  Error("MachSoftmaxStable::Forw CPU implementation not done!");
  REAL *optr, sum;
  int b=eff_bsize*odim;
     // apply exp() on all outputs
  VEXP(&b,data_out);
  for (b=0,optr=data_out; b<eff_bsize; b++,optr+=odim) {
    sum=1.0/ASUM(&odim,optr,&inc1);  // exp(x) is always positive -> we can use the sum_i (ABS(x_i))
    SCAL(&odim,&sum,optr,&inc1);
  }
#endif

  tmn.stop();
}

void MachSoftmaxStable::Backw(const float lrate, const float wdecay, int eff_bsize)
{
    // derivate softmax activation function
    //   do_i / da_k = o_i (kronecker_ik - o_k)
    // we suppose that do_i/da_k vanishes in the error function !!
    //             = o_i (1 - o_i)

   // this can't be done here since the result depends
   // on the error function (we must derivate each output w/r
   // to ALL other outputs. This can't be stored in one vector)
   //   dE/da_i = sum_k dE/do_k do_k/da_i
   // On the other hand, many terms vanish with usual error functions

  MachLin::Backw(lrate, wdecay, eff_bsize);
}

