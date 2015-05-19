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
 * $Id: MachLin.cpp,v 1.65 2014/03/26 06:03:25 schwenk Exp $
 */

using namespace std;
#include <iostream>
#include <stdlib.h>

#include "Tools.h"
#include "MachLin.h"
#include "Blas.h"
#ifdef CUDA
#  include "Gpu.cuh"
#endif

MachLin::MachLin(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw)
 : Mach(p_idim, p_odim, p_bsize, p_nbfw, p_nbbw), bw_shared(NULL), bw_mutex(NULL)
{
#ifdef BLAS_CUDA
  b = cuda_alloc(odim, "bias of linear machine");
  w = cuda_alloc(idim*odim, "weights of linear machine");
#else
  if (odim>0) {
    b = new REAL[odim];
    if (!b) Error ("can't allocate memory for bias of linear machine");
  }
  else b=NULL;
  if (idim*odim>0) {
    w = new REAL[idim*odim];
    if (!w) Error ("can't allocate memory for weights of linear machine");
  }
  else w=NULL;
#endif

    // biases and weights sharing
  bw_mutex = new pthread_mutex_t;
  if (bw_mutex != NULL) {
    pthread_mutex_init(bw_mutex, NULL);
    int *new_bw_shared = new int;
    if (new_bw_shared != NULL) {
      (*new_bw_shared) = 0;
      bw_shared = new_bw_shared;
    }
  }
}

MachLin::MachLin(const MachLin &m)
 : Mach(m), b(NULL), w(NULL), bw_shared(NULL), bw_mutex(NULL)
{
  int inc_bw_shared = 0;
  if (m.bw_mutex != NULL) {
    pthread_mutex_lock(m.bw_mutex);
    inc_bw_shared = ((m.bw_shared != NULL) ? (*m.bw_shared) + 1 : 0);
    if (inc_bw_shared > 0) {
      (*m.bw_shared) = inc_bw_shared;

        // share the weights and biases
      b = m.b;
      w = m.w;
      bw_shared = m.bw_shared;
      bw_mutex = m.bw_mutex;
    }
    pthread_mutex_unlock(m.bw_mutex);
  }
  if (inc_bw_shared <= 0)
    Error ("can't share memory for bias and weights of linear machine");
}

/*******************************************
 *
 ********************************************/

MachLin::~MachLin()
{

#ifdef BLAS_CUDA
#else
#if 0
  printf("W:\n");
  for (int od=0;od<odim;od++) {
    for (int id=0;id<idim;id++) printf(" %9.7f",w[id*odim+od]);
    printf("\n");
  }
  printf("b: ");
  for (int od=0;od<odim;od++) printf(" %9.7f",b[od]);
  printf("\n");
#endif
#endif

    // verify if biases and weights are shared
  if (bw_mutex != NULL) {
    pthread_mutex_lock(bw_mutex);
    if (bw_shared != NULL) {
      if ((*bw_shared) > 0) {
        (*bw_shared)--;
        pthread_mutex_unlock(bw_mutex);
        return;
      }
      else {
        delete bw_shared;
        bw_shared = NULL;
      }
    }
  }

#ifdef BLAS_CUDA
  if (b) cublasFree(b);
  if (w) cublasFree(w);
#else
  if (b) delete [] b;
  if (w) delete [] w;
#endif
  b = w = NULL;

    // destroy mutex
  if (bw_mutex != NULL) {
    pthread_mutex_t *old_bw_mutex = bw_mutex;
    bw_mutex = NULL;
    pthread_mutex_unlock(old_bw_mutex);
    pthread_mutex_destroy(old_bw_mutex);
    delete old_bw_mutex;
  }
}

void MachLin::BiasConst(const REAL val)
{
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
  Error("MachLin::BiasConst(): not implemented for CUDA\n");
#else
  for (int i=0; i<odim; i++) b[i]=val;
#endif
}

void MachLin::BiasRandom(const REAL range)
{
  REAL c=range*2.0;
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
#ifdef CURAND
  curandGenerateUniform(cuda_gen, (float*) b, odim);		// in (0,1]
  cuda_check_error("generating random values for biases");
  nppsSubC_32f_I(0.5,b,odim);
  nppsMulC_32f_I(c,b,odim);
#else
  REAL * tmp = new REAL[odim];
  for (int i=0; i<odim; i++) tmp[i]=c*(drand48()-0.5);
  cublasSetVector(odim, sizeof(REAL), tmp, 1, b, 1);  
  free(tmp);
#endif
#else
  for (int i=0; i<odim; i++) b[i]=c*(drand48()-0.5);
#endif
}

void MachLin::WeightsConst(const REAL val)
{
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
  Error("MachLin::WeightsConst(): not implemented for CUDA\n");
#else
  for (int i=0; i<idim*odim; i++) w[i]=val;
#endif
}

void MachLin::WeightsID(const REAL scale)
{
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
  REAL * tmp = new REAL[idim * odim];
  memset(tmp, 0.0, idim*odim*sizeof(REAL));
  if (idim>odim)
    for (int x=0; x<odim; x++) tmp[x*odim+x]=scale;
  else
    for (int x=0; x<idim; x++) tmp[x*odim+x]=scale;
  cublasSetVector(idim * odim, sizeof(REAL), tmp, 1, w, 1);
  delete [] tmp;
#else
  memset(w, 0.0, idim*odim*sizeof(REAL));
  if (idim>odim) 
    for (int x=0; x<odim; x++) w[x*odim+x]=scale;
  else
    for (int x=0; x<idim; x++) w[x*odim+x]=scale;
#endif
}

void MachLin::WeightsRandom(const REAL range)
{
  REAL c=range*2.0;
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
#ifdef CURAND
  curandGenerateUniform(cuda_gen, (float*) w, idim*odim);
  cuda_check_error("generating random values for biases");
  nppsSubC_32f_I(0.5,w,idim*odim);
  nppsMulC_32f_I(c,w,idim*odim);
#else
  REAL * tmp = new REAL[idim * odim];
  for (int i=0; i<idim * odim; i++) tmp[i]=c*(drand48()-0.5);
  cublasSetVector(idim * odim, sizeof(REAL), tmp, 1, w, 1);
  delete [] tmp;
#endif
#else
  for (int i=0; i<idim*odim; i++) w[i]=c*(drand48()-0.5);
#endif
}

void MachLin::WeightsRandomFanI(const REAL range)
{
  REAL c=2.0*range/sqrt((REAL) idim);
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
#ifdef CURAND
  curandGenerateUniform(cuda_gen, (float*) w, idim*odim);
  cuda_check_error("generating FanI random values for biases");
  nppsSubC_32f_I(0.5,w,idim*odim);
  nppsMulC_32f_I(c,w,idim*odim);
#else
  REAL * tmp = new REAL[idim * odim];
  for (int i=0; i<idim * odim; i++) tmp[i]=c*(drand48()-0.5);
  cublasSetVector(idim * odim, sizeof(REAL), tmp, 1, w, 1);
  delete [] tmp;
#endif
#else
  printf("weight init FanI=%d, range =%5.3e\n",idim, c/2.0);
  for (int i=0; i<idim*odim; i++) w[i]=c*(drand48()-0.5);
#endif
}

void MachLin::WeightsRandomFanIO(const REAL range)
{
  REAL c=2.0*range/sqrt((REAL) (idim+odim));
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
#ifdef CURAND
  curandGenerateUniform(cuda_gen, (float*) w, idim*odim);
  cuda_check_error("generating FanIO random values for biases");
  nppsSubC_32f_I(0.5,w,idim*odim);
  nppsMulC_32f_I(c,w,idim*odim);
#else
  REAL * tmp = new REAL[idim * odim];
  for (int i=0; i<idim * odim; i++) tmp[i]=c*(drand48()-0.5);
  cublasSetVector(idim * odim, sizeof(REAL), tmp, 1, w, 1);
  delete [] tmp;
#endif
#else
  for (int i=0; i<idim*odim; i++) w[i]=c*(drand48()-0.5);
#endif
}

void MachLin::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on linear machine" << endl;
    Mach::Info(detailed,txt);
  }
  else {
#ifdef BLAS_CUDA
    printf(", on GPU %d", cuda_dev);
#endif
    tm.disp(", ");
    tm.newline();

#ifdef BLAS_CUDA
#else
#endif
  }
}

//-----------------------------------------------
// File output
//-----------------------------------------------

void MachLin::WriteData(ofstream &outf)
{
  int s=odim*idim + odim;
  outf.write((char*) &s,sizeof(int));
  s=sizeof(REAL);
  outf.write((char*) &s,sizeof(int));

#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
  REAL *local_mem=new REAL[odim*idim];
  cublasGetVector(odim*idim,CUDA_SIZE,w,1,local_mem,1);
  cuda_check_error("transfer of weight matrix from GPU memory");
  outf.write((char*)local_mem,odim*idim*sizeof(REAL));
  delete [] local_mem;

  local_mem=new REAL[odim];
  cublasGetVector(odim,CUDA_SIZE,b,1,local_mem,1);
  cuda_check_error("transfer of bias vector from GPU memory");
  outf.write((char*)local_mem,odim*sizeof(REAL));
  delete [] local_mem;
#else
  outf.write((char*) w,odim*idim*sizeof(REAL));
  outf.write((char*) b,odim*sizeof(REAL));
#endif
}

//-----------------------------------------------
// File input
//-----------------------------------------------


void MachLin::ReadData(ifstream &inpf, size_t s, int bs)
{
  size_t se=odim*idim + odim;
  if (s!=se) {
    cerr << "ERROR: data block of linear machine has " << s << " elements (" << se << " were expected)" << endl;    Error();
  } 
  Mach::ReadData(inpf, 0, bs);

    // read parameters
    // TODO: error checks
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
  REAL *local_mem=new REAL[odim*idim];
  inpf.read((char*)local_mem,odim*idim*sizeof(REAL));
  for (int i=0;i<idim*odim;i++) 
    if (isnan(local_mem[i])) ErrorN("NAN in weights of layer %dx%d\n",idim,odim);
  cublasSetVector(odim*idim,CUDA_SIZE,local_mem,1,w,1);
  cuda_check_error("transfer of weight matrix to GPU memory");
  delete [] local_mem;

  local_mem=new REAL[odim];
  inpf.read((char*)local_mem,odim*sizeof(REAL));
  for (int i=0;i<odim;i++) 
    if (isnan(local_mem[i])) ErrorN("NAN in bias of layer %dx%d\n",idim,odim);
  cublasSetVector(odim,CUDA_SIZE,local_mem,1,b,1);
  cuda_check_error("transfer of bias vector to GPU memory");
  delete [] local_mem;
#else
  inpf.read((char*) w,odim*idim*sizeof(REAL));
  inpf.read((char*) b,odim*sizeof(REAL));
    // checking for bad values
  for (int i=0;i<idim*odim;i++) 
    if (isnan(w[i])) ErrorN("NAN in weights of layer %dx%d\n",idim,odim);
  for (int i=0;i<odim;i++) 
    if (isnan(w[i])) ErrorN("NAN in bias of layer %dx%d\n",idim,odim);
#if 0
cout << "\nRead from file:" << endl;
  printf("W: %dx%d\n",odim,idim);
  for (int od=0;od<odim;od++) {
    for (int id=0;id<idim;id++) printf(" %9.7f",w[id*odim+od]);
    printf("\n");
  }
  printf("b:\n");
  for (int od=0;od<odim;od++) printf(" %9.7f",b[od]);
  printf("\n");
#endif
#endif
}


//-----------------------------------------------
// Training
//-----------------------------------------------

void MachLin::Forw(int eff_bsize)
{

  tm.start();

  if (!data_in)
    Error("MachLin::Forw(): input data is not set");
  if (eff_bsize<=0) eff_bsize=bsize;

  debugMachInp("MachLin",data_in,idim,odim,eff_bsize);

#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
    // copy bias <eff_bsize> times into result matrix 
  GpuCopyVectorToMatrix(data_out, b, eff_bsize, odim);
  call_gemm(data_out, w, data_in, 1.0, odim, eff_bsize, idim);
#else
  for (int e=0; e<eff_bsize; e++)
    memcpy(data_out+e*odim,b,odim*sizeof(REAL));
  call_gemm(data_out, w, data_in, 1.0, odim, eff_bsize, idim);
#endif
  nb_forw += eff_bsize;

  tm.stop();
  debugMachOutp("MachLin",data_out,idim,odim,eff_bsize);
}


void MachLin::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  static REAL real1=1.0, real0=0.0;
  static char transN='N', transT='T';
  REAL lrate_bs = lrate / sqrt(GetBsize());	// scale by block size !
  REAL epsilon = 1.0 + lrate_bs * wdecay;

  if (eff_bsize<=0) eff_bsize=bsize;
  if (!grad_out)
    Error("MachLin::Backw(): output gradient is not set");

  debugMachOutp("MachLin Grad",grad_out,idim,odim,eff_bsize);
  tm.start();

#if defined(BLAS_ATLAS) || defined(BLAS_INTEL_MKL)
    // perform drop-out
  if (drop_out>0.0) {
    REAL coef=1.0/(1.0-drop_out);
    REAL *rptr=drop_out_rand;
    REAL *gptr=grad_out;
    for (int i=0; i<eff_bsize*odim; i++) {
      if (*rptr++<drop_out) *gptr++ = 0.0;
                       else *gptr++ *= coef;
    }
  }

    // update bias vector:   b = b + lrate * grad_out
    // NO weight decay
  REAL *gptr = grad_out;
  for (int e=0; e<eff_bsize; e++, gptr+=odim) {
    AXPY(&odim,&lrate_bs,gptr,&inc1,b,&inc1);
  }

    // backprop gradient:   grad_in   =        w'        *   grad_out
    //                    idim x bsize = (odim x idim)'  *  odim x bsize
  GEMM (&transT, &transN, &idim, &eff_bsize, &odim,
        &real1, w, &odim, grad_out, &odim,
        &real0, grad_in, &idim);

    // update weights including weight decay
    // w = lrate  *grad_out * data_in^T + epsilon * w
    // gemm (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
    //                                      Go      Din            W
    //        C = alpha*A * B + beta * b
    //

  GEMM (&transN, &transT, &odim, &idim, &eff_bsize,
        &lrate_bs, grad_out, &odim, data_in, &idim,
        &epsilon, w, &odim);
#else
# ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);

    // perform drop-out
  if (drop_out>0) {
    GpuDropOut(odim*eff_bsize, grad_out, drop_out_rand, drop_out);
  }

  GpuBatchedAXPY(odim,lrate_bs,grad_out,1,b,1,eff_bsize);
    // backprop gradient:   grad_in   =        w'        *   grad_out
    //                    idim x bsize = (odim x idim)'  *  odim x bsize
  GEMM (transT, transN, idim, eff_bsize, odim,
        real1, w, odim, grad_out, odim,
        real0, grad_in, idim);
    // update weights including weight decay
    // w = lrate  *grad_out * data_in^T + epsilon * w
  GEMM (transN, transT, odim, idim, eff_bsize,
        lrate_bs, grad_out, odim, data_in, idim,
        epsilon, w, odim);
# else
  Error("you must compile with BLAS_ATLAS, BLAS_INTEL_MKL or BLAS_CUDA");
# endif
#endif
  nb_backw += eff_bsize;

  tm.stop();
  debugMachInp("MachLin Grad",grad_in,idim,odim,eff_bsize);
}

void MachLin::Debug()
{
#ifdef BLAS_CUDA
  Error("MachLin::Debug(): not implemented for CUDA\n");
#else
  for (int o=0; o<odim; o++) {
    for (int i=0; i<idim; i++) {
      w[i*odim+o] = i + 1000*o;
    }
    b[o] = -o;
  }
#endif
}
