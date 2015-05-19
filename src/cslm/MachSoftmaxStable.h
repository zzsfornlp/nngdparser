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
 * $Id: MachSoftmaxStable.h,v 1.7 2014/03/25 21:52:53 schwenk Exp $
 *
 * softmax machine:  a_i = exp(a_i) / sum_k a_k
 * The stable version remove the max of a_i to all a_i before exp.
 *     This lower numerical precision problem
 * with a_k is the kth output of a linear machine
 */

#ifndef _MachSoftmaxStable_h
#define _MachSoftmaxStable_h

#include "MachLin.h"

#undef BLAS_CUDA_NPPS_SUM	// thsi should be faster, but I can't get it working

class MachSoftmaxStable : public MachLin
{
private:
  Timer tmn;				// cumulated time used for softmax normalization
#if defined(BLAS_CUDA) && defined(BLAS_CUDA_NPPS_SUM)
  Npp8u *gpu_sum_buf;			// temporary buffer for fast sum with nppsSum_32f()
#endif
protected:
  MachSoftmaxStable(const MachSoftmaxStable &);			// create a copy of the machine
public:
  MachSoftmaxStable(const int=0, const int=0, const int=128, const ulong=0, const ulong=0);	
  virtual ~MachSoftmaxStable();
  virtual MachSoftmaxStable *Clone() {return new MachSoftmaxStable(*this);}	// create a copy of the machine
  virtual int GetMType() {return file_header_mtype_softmax_stable;};	// get type of machine
  virtual void Info(bool=false, char *txt=(char*)"");	// display (detailed) information on machine
  virtual void Forw(int=0);	// calculate outputs for current inputs
    // backprop gradients from output to input and update all weights
  virtual void Backw (const float lrate, const float wdecay, int=0);
};

#endif
