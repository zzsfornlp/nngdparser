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
 * $Id: MachTab.cpp,v 1.51 2014/03/25 21:52:53 schwenk Exp $
 */

using namespace std;
#include <iostream>
#include <stdlib.h>

#include "Tools.h"
#include "MachTab.h"
#include "Blas.h"

#ifdef BLAS_CUDA
# include "Gpu.cuh"
#endif

void MachTab::do_alloc()
{
  if (!ext_table) {
#ifdef BLAS_CUDA
    cudaSetDevice(cuda_dev);
    t = cuda_alloc(idim*odim, "memory for table look-up machine");
#else
    t = new REAL[idim*odim];
    if (!t) Error ("can't allocate memory for table look-up machine");
#endif
  }
  else
    ;
#ifdef BLAS_CUDA
    tmp_inp = new REAL[idim*bsize];
#endif
}


MachTab::MachTab(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw)
 : Mach(1, p_odim, p_bsize, p_nbfw, p_nbbw), ext_table(false), t_shared(NULL), t_mutex(NULL)
{
  if (p_idim<=0) Error("Table machine: illegal value of input dimension");
  if (p_odim<=0) Error("Table machine: illegal value of output dimension");
  idim = p_idim; // override 1 in call to Mach()

  do_alloc();

    // look-up table sharing
  t_mutex = new pthread_mutex_t;
  if (t_mutex != NULL) {
    pthread_mutex_init(t_mutex, NULL);
    int *new_t_shared = new int;
    if (new_t_shared != NULL) {
      (*new_t_shared) = 0;
      t_shared = new_t_shared;
    }
  }
}

MachTab::MachTab(REAL *table,
	const int p_idim, const int p_odim, const int p_bsize,
	const ulong p_nbfw, const ulong p_nbbw)
 : Mach(1, p_odim, p_bsize, p_nbfw, p_nbbw), ext_table(true),
   t_shared(NULL), t_mutex(NULL)
{
  if (p_idim<0) Error("Table machine: illegal value of input dimension");
  if (p_odim<0) Error("Table machine: illegal value of output dimension");
  idim = p_idim; // override 1 in call to Mach()

  //if (!table) Error ("Table look-up machine: provided address is NULL");
  t=table;
  do_alloc();
}

MachTab::MachTab(const MachTab &m)
 : Mach(m, 1), ext_table(true), t(NULL),
   t_shared(NULL), t_mutex(NULL)
{
  idim = m.idim; // override 1 in call to Mach()
  ext_table = m.ext_table;

  if (ext_table) {
    // set look-up table with external address
    t = m.t;
    do_alloc();
  }
  else {
    int inc_t_shared = 0;
    if (m.t_mutex != NULL) {
      pthread_mutex_lock(m.t_mutex);
      inc_t_shared = ((m.t_shared != NULL) ? (*m.t_shared) + 1 : 0);
      if (inc_t_shared > 0) {
        (*m.t_shared) = inc_t_shared;

          // share the look-up table
        t = m.t;
        t_shared = m.t_shared;
        t_mutex = m.t_mutex;
      }
      pthread_mutex_unlock(m.t_mutex);
    }
    if (inc_t_shared <= 0)
      Error ("can't share memory for table look-up machine");
  }
}

MachTab::~MachTab()
{

#ifdef BLAS_CUDA
  if (tmp_inp) delete tmp_inp;
#endif

    // verify if the look-up table is shared
  if (t_mutex != NULL) {
    pthread_mutex_lock(t_mutex);
    if (t_shared != NULL) {
      if ((*t_shared) > 0) {
        (*t_shared)--;
        pthread_mutex_unlock(t_mutex);
        return;
      }
      else {
        delete t_shared;
        t_shared = NULL;
      }
    }
  }

#ifdef BLAS_CUDA
  if (!ext_table & (t!=NULL)) cublasFree(t);
#else
  if (!ext_table & (t!=NULL)) delete [] t;
#endif
  t = NULL;

    // destroy mutex
  if (t_mutex != NULL) {
    pthread_mutex_t *old_t_mutex = t_mutex;
    t_mutex = NULL;
    pthread_mutex_unlock(old_t_mutex);
    pthread_mutex_destroy(old_t_mutex);
    delete old_t_mutex;
  }
}

void MachTab::TableConst(const REAL val)
{
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
  nppsSet_32f(val,t,idim*odim);
#else
  for (int i=0; i<idim*odim; i++) t[i]=val;
#endif
}

void MachTab::TableRandom(const REAL range)
{
  REAL c=range*2.0;
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
#ifdef CURAND
  curandGenerateUniform(cuda_gen, (float*) t, idim*odim);
  cuda_check_error("generating random values for table look-up machine");
  nppsSubC_32f_I(0.5,t,idim*odim);
  nppsMulC_32f_I(c,t,idim*odim);
#else
  REAL * tmp = new REAL[idim * odim];
  for (int i=0; i<idim * odim; i++) tmp[i]=c*(drand48()-0.5);
  cublasSetVector(idim * odim, sizeof(REAL), tmp, 1, t, 1);
  delete [] tmp;
#endif
#else
  for (int i=0; i<idim*odim; i++) t[i]=c*(drand48()-0.5);
#endif
}

void MachTab::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on table look-up machine" << endl;
    Mach::Info(detailed,txt);
  }
  else {
    printf("%sMachTab %c[%d]-%d, bs=%d, passes=%lu/%lu", txt, ext_table ? 's' : '1', idim, odim, bsize, nb_forw, nb_backw);
#ifdef BLAS_CUDA
    printf(", on GPU %d", cuda_dev);
#endif
    tm.disp(", ");
    printf("\n");
  }
}

//-----------------------------------------------
// File output
//-----------------------------------------------

void MachTab::WriteParams(ofstream &of)
{

  Mach::WriteParams(of);
  of.write((char*) &ext_table, sizeof(int));
}


void MachTab::WriteData(ofstream &outf) {
  int i=0, s=sizeof(REAL);
  if (ext_table) {
    outf.write((char*) &i, sizeof(int));
    outf.write((char*) &s, sizeof(int));
  }
  else {
    i=idim*odim;
    outf.write((char*) &i, sizeof(int));
    outf.write((char*) &s, sizeof(int));
#ifdef BLAS_CUDA
    REAL *local_mem=new REAL[i];
    cudaSetDevice(cuda_dev);
    cublasGetVector(i,CUDA_SIZE,t,1,local_mem,1);
    cuda_check_error("transfer of table look-up machine from GPU memory");
    outf.write((char*)local_mem,i*sizeof(REAL));
    delete [] local_mem;
#else
    outf.write((char*) t,i*sizeof(REAL));
#endif

  }
}

//-----------------------------------------------
// File input
//-----------------------------------------------

void MachTab::ReadParams(ifstream &inpf, bool with_alloc)
{

  Mach::ReadParams(inpf, false);
  inpf.read((char*) &ext_table, sizeof(int));
  do_alloc();
}

void MachTab::ReadData(ifstream &inpf, size_t s, int bs)
{
  size_t se=odim*idim;

  if (ext_table) {
    if (s>0) {
      ErrorN("internal error in file, table look-up machine has external address, but %u elements of data are provided\n",(uint)s);
    }
    return;	// address will be filled in by MachPar
  }
  else if (s!=se) {
    ErrorN("data block of table look-up machine has %u elements - %u were expected)",(uint) s, (uint) se);
  } 
  Mach::ReadData(inpf, 0, bs);
#ifdef BLAS_CUDA
  REAL *local_mem=new REAL[odim*idim];
  inpf.read((char*)local_mem,odim*idim*sizeof(REAL));
  cudaSetDevice(cuda_dev);
  cublasSetVector(odim*idim,CUDA_SIZE,local_mem,1,t,1);
  cuda_check_error("transfer of table look-up machine to GPU memory");
  delete [] local_mem;
#else
  inpf.read((char*) t,odim*idim*sizeof(REAL));
#endif
}


//-----------------------------------------------
// Training
//-----------------------------------------------

void MachTab::Forw(int eff_bsize)
{
  if (!data_in)
    Error("MachTab::Forw(): input data is not set");

  debugMachInp("MachTab",data_in,1,odim,eff_bsize);

#if 0
  printf("CODES: %d%%d\n",idim,odim);
  REAL *tptr=t;
  for (int i=0; i<idim; i++) {
    printf("code %2d:", i);
    for (int o=0; o<odim; o++) printf(" %5.2f", *tptr++);
    printf("\n");
  }
#endif

  tm.start();

  if (eff_bsize<=0) eff_bsize=bsize;

#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
  GpuMachTabForw(eff_bsize, odim, data_in, t, data_out);
#else
  REAL *optr=data_out;
  for (int b=0; b<eff_bsize; b++) {
    int idx= (int) data_in[b];
    if (idx==NULL_WORD) {
        // simulate empty word: set everything to 0
      for (int i=0; i<odim; i++) *optr++=0.0;
    }
    else {
      memcpy(optr,t+idx*odim,odim*sizeof(REAL));
      optr+=odim;
    }
  }
#endif

  nb_forw+=eff_bsize;

  tm.stop();
  debugMachOutp("MachTab",data_out,idim,odim,eff_bsize);
}


void MachTab::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  // table[wid] = table[wid] + lrate * grad_out[wid] * data_in[wid]

  REAL lrate_bs = lrate / sqrt(GetBsize());	// scale by block size !
  tm.start();

#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
  GpuMachTabBackw(lrate_bs,eff_bsize, odim, data_in, t, grad_out);
    // we don't backprop to the input of a table look-up machine
  nppsSet_32f(0.0,grad_in,eff_bsize);
#else
  REAL *gptr = grad_out;
  for (int b=0; b<eff_bsize; b++,gptr+=odim) {
    int idx= (int) data_in[b];
    if (idx==NULL_WORD) { // empty word: no weight update
    }
    else {
      REAL *tptr=t+idx*odim;
      AXPY(&odim,&lrate_bs,gptr,&inc1,tptr,&inc1);
    }
  }
 
    // we don't backprop to the input of a table look-up machine
  for (int b=0; b<eff_bsize; b++) grad_in[b]=0.0;
#endif

  tm.stop();
}

