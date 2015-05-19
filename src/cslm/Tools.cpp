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
 * $Id: Tools.cpp,v 1.15 2014/03/25 21:52:53 schwenk Exp $
 */

#include <iostream>
#include <signal.h>
#include <cstdarg>

#include "Tools.h"
#ifdef BLAS_CUDA
# include <cuda_runtime_api.h>
#endif

extern void GpuUnlock();	// forward declaration

void Error(void)
{
  GpuUnlock();
  exit(1);
}

//******************************************

void Error(const char *txt)
{
  ErrorN(txt);
}

//******************************************

void Error(const char *txt, int ipar)
{
  fprintf(stderr,"ERROR: ");
  fprintf(stderr,txt,ipar);
  fprintf(stderr,"\n");
  GpuUnlock();
  exit(1);
}

//******************************************

__attribute__((noreturn))
void vErrorN(const char* msg, va_list args)
{
    char message[ERROR_MSG_SIZE];

#if !defined(ULTRIX) && !defined(_MINGW_) && !defined(WIN32)
    vsnprintf(message, ERROR_MSG_SIZE,msg,args);
#else
    vsprintf(message,msg,args);
#endif
    cerr <<" ERROR: "<<message<<endl;
#ifdef BLAS_CUDA
    int dev;
    cudaGetDevice(&dev);
    cerr <<" Current GPU device: "<<dev<<endl;
#endif
    GpuUnlock();
    exit(1);
}
     
//******************************************

void ErrorN(const char* msg, ...){
    va_list args;
    va_start(args,msg);
    vErrorN(msg, args);
    va_end(args);
}

//******************************************

int ReadInt(ifstream &inpf, const string &name, int minval,int maxval)
{
  string buf;
  inpf >> buf;
  if (buf!=name) {
    cerr << "FileRead: found field '" << buf << "' while looking for '" << name << "'";
    Error("");
  }
    
  int val;
  inpf >> val;
  if (val<minval || val>maxval) {
    cerr << "FileRead: values for " << name << "must be in ["<<minval<<","<<maxval<<"]";
    Error("");
  }

  return val;
}

//******************************************

#ifdef BLAS_CUDA
void DebugMachInp(string txt, REAL *iptr, int idim, int odim, int eff_bsize) {
  Error("debugging of input data not supported for CUDA"); }
void DebugMachOutp(string txt, REAL *optr, int idim, int odim, int eff_bsize) {
  Error("debugging of output data not supported for CUDA"); }
#else
void DebugMachInp(string txt, REAL *iptr, int idim, int odim, int eff_bsize)
{
  cout <<"\n" << txt;
  printf(" %dx%d bs=%d: input\n",idim,odim,eff_bsize);
  for (int bs=0;bs<eff_bsize;bs++) {
    printf("%3d:  ",bs);
    for (int i=0;i<idim;i++) printf(" %4.2f", *iptr++);
    printf("\n");
  }
}

//******************************************

void DebugMachOutp(string txt, REAL *optr, int idim, int odim, int eff_bsize)
{
  cout <<"\n" << txt;
  printf(" %dx%d bs=%d: output\n",idim,odim,eff_bsize);
  for (int bs=0;bs<eff_bsize;bs++) {
    printf("%3d:  ",bs);
    for (int o=0;o<odim;o++) printf(" %4.2f", *optr++);
    printf("\n");
  }
}
#endif
