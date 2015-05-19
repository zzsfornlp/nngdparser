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
 * $Id: MachLinRectif.cpp,v 1.7 2014/03/25 21:52:53 schwenk Exp $
 */

using namespace std;
#include <iostream>
#include <math.h>

#include "Tools.h"
#include "MachLinRectif.h"
#include "Blas.h"
#include "Gpu.cuh"

MachLinRectif::MachLinRectif(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw)
 : MachLin(p_idim, p_odim, p_bsize, p_nbfw, p_nbbw)
{
}

MachLinRectif::MachLinRectif(const MachLinRectif &m)
 : MachLin(m)
{
}

MachLinRectif::~MachLinRectif()
{
}


//-----------------------------------------------
// Tools
//-----------------------------------------------

void MachLinRectif::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on linear rectifier machine" << endl;
    MachLin::Info(detailed,txt);
  }
  else {
    printf("%sMachLinRectif %d-%d, bs=%d, passes=%lu/%lu", txt, idim, odim, bsize, nb_forw, nb_backw);
#ifdef BLAS_CUDA
    printf(", on GPU %d", cuda_dev);
#endif
    tm.disp(", ");
    tmh.disp(" + recif: ");
    printf("\n");
  }
}

//-----------------------------------------------
// Training
//-----------------------------------------------

void MachLinRectif::Forw(int eff_bsize)
{

  if (eff_bsize<=0) eff_bsize=bsize;
  MachLin::Forw(eff_bsize);

  tmh.start();

    // apply linear rectifier on output
#ifdef BLAS_CUDA
  GpuLinRectifForw(odim*eff_bsize, data_out);
#else
  REAL *ptr=data_out;
  for (int i=0; i<odim*eff_bsize; i++, ptr++) {
    if (*ptr<0) *ptr=0;
  }
#endif

  tmh.stop();
}

void MachLinRectif::Backw(const float lrate, const float wdecay, int eff_bsize)
{
    // derivate tanh activation function
    // multiply grad_hidden by derivatives of hidden layer activities (tanh)
    // grad_out = grad_out .* f'(data_out)
    //          = grad_out .* ( 1 - data_out^2 )

  if (eff_bsize<=0) eff_bsize=bsize;
  if (!grad_out)
    Error("MachLinRectif::Backw(): output gradient is not set");

  tmh.start();
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
  GpuLinRectifBackw(odim*eff_bsize, data_out, grad_out);
#else
  REAL *dptr=data_out;
  REAL *gptr=grad_out;
  for (int i=0; i<odim*eff_bsize; i++) {
    if (*dptr++<0) *gptr++=0;  // multiply by 0
               else gptr++; // multiply by 1
  }
#endif
  tmh.stop();

  MachLin::Backw(lrate, wdecay, eff_bsize);
}

