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
 * $Id: Mach.cpp,v 1.63 2014/03/25 21:52:53 schwenk Exp $
 */

using namespace std;
#include <iostream>
#include <signal.h>
#include <unistd.h>

#include "Tools.h"
#include "Mach.h"
#include "MachTab.h"
#include "MachLin.h"
#include "MachSig.h"
#include "MachTanh.h"
#include "MachSoftmax.h"
#include "MachSoftmaxStable.h"
#include "MachLinRectif.h"
#include "MachSeq.h"
#include "MachPar.h"
#include "MachSplit.h"
#include "MachSplit1.h"

vector<Mach*> signal_mach;
static int fileid=-1;

#ifdef BLAS_CUDA
# include "Blas.h"
  // global variables
cublasStatus cuda_stat;
curandGenerator_t cuda_gen;
vector<int> cuda_devs;	// user specified list of GPUs to be used

//
//

#define LOCK_FNAME "/tmp/gpu_lock.pid%d.gpu%d"
#define LOCK_FNAME_LEN 256	// Hack ;-)

void HandlerSigTERM(int s)
{
  printf("Catched signal: removing lock-files\n");
  GpuUnlock();
  exit(1);
}

//
//


void cuda_init()
{
  static int cuda_init_status=-1;
  struct cudaDeviceProp props;

  if (cuda_init_status>=0) return;

  cout << "Initializing Nvidia GPU card" << endl;

  int n, d;
  cudaGetDeviceCount(&n);
  if (n>1) {
    cout << " - found " << n << " cards:" << endl;
    for (d=0; d<n; d++) {
      cudaGetDeviceProperties(&props, d);
      int nb_cores_per_multiprocessor = -1;
      if(props.major == 1 && (props.minor == 0||props.minor == 1||props.minor == 2||props.minor == 3))
          nb_cores_per_multiprocessor = 8;
      else if(props.major == 2 && props.minor == 0)
          nb_cores_per_multiprocessor = 32;
      else if(props.major == 2 && props.minor == 1)
          nb_cores_per_multiprocessor = 48;
      else if(props.major == 3 && (props.minor == 0||props.minor == 5))
          nb_cores_per_multiprocessor = 192;


      printf("    %d: %s with %d CPUs x %d threads running at %4.2f Ghz, %d MBytes of memory, use -arch=sm_%d%d\n",
	  d, props.name, props.multiProcessorCount, nb_cores_per_multiprocessor,
          props.clockRate/1000000.0, (int) (props.totalGlobalMem/1024/1024),
          props.major, props.minor);
    }
  }

  switch (cuda_devs.size()) {
    case 0: printf(" - no GPU device specified, using default device 0\n");
            cuda_devs.push_back(0);
    case 1: printf(" - using device %d\n", cuda_devs[0]);
            cudaSetDevice(cuda_devs[0]);
            break;
    default:
      if (cuda_devs.size()>(uint)n) {
        printf(" - requested more GPU devices than available, using %d first ones\n", n);
        for (uint ii=n; ii<cuda_devs.size(); ii++) cuda_devs.pop_back();
      }
      printf(" - using %lu devices in parallel:", cuda_devs.size());
      for (uint ii=0; ii<cuda_devs.size(); ii++) {
        printf(" %d", cuda_devs[ii]);
        if (cuda_devs[ii]<0 || cuda_devs[ii]>=n) Error("illegal device identifier");
      }
      printf("\n");
      cudaSetDevice(cuda_devs[0]);
  }

    // initialize cublas and random generator
  cublasInit();
  cuda_check_error("initialization of card\n");
  curandCreateGenerator(&cuda_gen, CURAND_RNG_PSEUDO_DEFAULT);
  cuda_check_error("initialization of random generator\n");
  cuda_init_status=0;

    // locking devices
  pid_t getpid(void);
  ofstream lfs;
  char lfname[LOCK_FNAME_LEN] = LOCK_FNAME;
  for (uint ii=0; ii<cuda_devs.size(); ii++) {
    sprintf(lfname,LOCK_FNAME,getpid(),cuda_devs[ii]);
    lfs.open(lfname,ios::out);
    CHECK_FILE(lfs,lfname);
    lfs << "Runing job " << getpid() << " on GPU " << cuda_devs[ii] << endl;
    lfs.close();
  }
   // catch signals to clean up lock-file
  signal(SIGINT, HandlerSigTERM);
  signal(SIGHUP, HandlerSigTERM);
  signal(SIGFPE, HandlerSigTERM);
  signal(SIGSEGV, HandlerSigTERM);
  signal(SIGTERM, HandlerSigTERM);
}

#else

int inc1=1;
#endif

void HandlerSigUSR1(int s) {
  time_t now;
  time(&now); // TODO: ctime is not rentrant ! use ctime_r() instead if needed
  cout << " - catched signal USR1 at " << ctime(&now) << endl;
  signal_mach[0]->Info(false, (char*)" -   ");
  cout.flush();
  //for (uint i=0; i<1; i++) signal_mach[i]->Info(false, (char*)" -   ");
  signal(SIGUSR1, HandlerSigUSR1);
}

//***********************************************

#ifdef BLAS_CUDA
void Mach::do_alloc()
{  
  cuda_init();

  data_out = cuda_alloc(odim*bsize, "output data for a machine");
  data_in=NULL; //  should be set later by SetDataIn()
  drop_out_rand = cuda_alloc(odim*bsize, "buffer for random values for drop-out");

  grad_in = cuda_alloc(idim*bsize, "input gradient for a machine");
  grad_out=NULL; // should be set later by SetGradOut()
}
#endif

//***********************************************

#ifndef BLAS_CUDA
void Mach::do_alloc()
{
  if (odim*bsize>0) {
    data_out=::new REAL[odim*bsize];
    if (!data_out) Error ("can't allocate memory for data_out");
    drop_out_rand=::new REAL[odim*bsize];
    if (!drop_out_rand) Error ("can't allocate memory for drop_out");
  }
  else { data_out=drop_out_rand=NULL; }
  data_in=NULL; // (luint) this) should be set later by SetDataIn() 
  if (idim*bsize>0) {
    grad_in=::new REAL[idim*bsize];
    if (!grad_in) Error ("can't allocate memory for grad_in");
  }
  else grad_in=NULL;
  grad_out=NULL; // (luint) this) should be set later by SetGradOut()
}
#endif


Mach::Mach(const int p_idim, const int p_odim, const int p_bsize, const ulong p_nbfw, const ulong p_nbbw)
 : idim(p_idim), odim(p_odim), bsize(p_bsize), nb_forw(p_nbfw), nb_backw(p_nbbw), drop_out(-1.0), drop_out_rand(NULL)
{
  do_alloc();
#ifdef BLAS_CUDA
  cudaGetDevice(&cuda_dev);
#endif

    // setup SIGUSR1 handler
  //cout << " - setting up handler for signal USR1" << endl;
  if (signal_mach.empty()) signal(SIGUSR1, HandlerSigUSR1);
  signal_mach.push_back(this);
}

Mach::Mach(const Mach &m, const int p_idim)
{
  if (p_idim > 0)
    idim = p_idim;
  else
    idim = m.idim;
  odim = m.odim;
  bsize = m.bsize;
  nb_forw = m.nb_forw;
  nb_backw = m.nb_backw;
  drop_out = m.drop_out;
  drop_out_rand = NULL;
#ifdef BLAS_CUDA
  cuda_dev = m.cuda_dev; // this is very important ! we share the weights so they must be on the same machine
  cudaSetDevice(cuda_dev);
#endif
  do_alloc();
  data_in = m.data_in;
  grad_out = m.grad_out;

    // setup SIGUSR1 handler
  //cout << " - setting up handler for signal USR1" << endl;
  if (signal_mach.empty()) signal(SIGUSR1, HandlerSigUSR1);
  signal_mach.push_back(this);
}

/*******************************************
 *
 ********************************************/

Mach::~Mach()
{
#ifdef BLAS_CUDA
  if (data_out) cublasFree(data_out);
  if (drop_out_rand) cublasFree(drop_out_rand);
  if (grad_in) cublasFree(grad_in);
#else
  if (data_out) delete [] data_out;
  if (drop_out_rand) delete [] drop_out_rand;
  if (grad_in) delete [] grad_in;
#endif
  signal_mach.pop_back();	 //TODO: we should search for correct machine and delete it
}

//-----------------------------------------------
// File output
//-----------------------------------------------

void Mach::WriteParams(ofstream &of) {
    // write machine specific params
  of.write((char*) &nb_forw, sizeof(ulong));
  of.write((char*) &nb_backw, sizeof(ulong));
}

void Mach::WriteData(ofstream &of) {
  const int i=0, s=sizeof(REAL);
  of.write((char*) &i, sizeof(int));
  of.write((char*) &s, sizeof(int));
}

void Mach::Write(ofstream &of)
{
  char header[file_header_size];
  for (int i=0; i<file_header_size; i++) header[i]=' ';
  sprintf(header,"%s %d",file_header_name, file_header_version);
  of.write(header,file_header_size);
  of.write((char*) &idim, sizeof(int));
  of.write((char*) &odim, sizeof(int));
  of.write((char*) &bsize, sizeof(int));
  int mtype=GetMType();
  of.write((char*) &mtype, sizeof(int));
  WriteParams(of);
  WriteData(of);
}

//-----------------------------------------------
// File input
//-----------------------------------------------


void Mach::ReadParams(ifstream &inpf, bool with_alloc)
{
  switch (fileid) {
    case file_header_version1: // read int but store ulong
      unsigned int itmp;
      inpf.read((char*) &itmp, sizeof(int)); nb_forw = (ulong) itmp;
      inpf.read((char*) &itmp, sizeof(int)); nb_backw = (ulong) itmp;
      break;
    case file_header_version2: 
      inpf.read((char*) &nb_forw, sizeof(ulong));
      inpf.read((char*) &nb_backw, sizeof(ulong));
      break;
    default:
      Error("internal error, fileid is unset");
  }
}

void Mach::ReadData(ifstream &inpf, size_t s, int bs)
{
  // there is nothing to read
}

Mach *Mach::Read(ifstream &inpf, int bs)
{
  char header[file_header_size], h[file_header_size];
  int v;

  inpf.read(header,file_header_size);
  if (sscanf(header,"%s %d",h,&v) != 2) {
    ErrorN("format of machine file not recognised: %s", header);
  }
  if (fileid<0) {
    fileid=v;
  }
  else {
    if (v!=fileid) ErrorN("all network files must have the same file ID %d",fileid);
  }
  if (strcmp(h,file_header_name)) {
    ErrorN("unsupported file type (%s), expected '%s'\n", h, file_header_name);
  }
  switch (fileid) {
    case file_header_version1: break;
    case file_header_version2: break;
    default:
      ErrorN("unsupported version of machine file (%d)\n",fileid);
  }

    // read idim, odim, bsize 
  int f_idim, f_odim, f_bsize;
  inpf.read((char*) &f_idim, sizeof(int));
  inpf.read((char*) &f_odim, sizeof(int));
  inpf.read((char*) &f_bsize, sizeof(int));
  if (bs <= 0)
    bs = f_bsize;

   // read and parse machine type
  int mtype;
  Mach *m=NULL;
  inpf.read((char*) &mtype, sizeof(int));
  switch (mtype) {
    case file_header_mtype_base: m = new Mach(f_idim,f_odim,bs); break;
    case file_header_mtype_tab: m = new MachTab(NULL,f_idim,f_odim,bs,0,0); break;
    case file_header_mtype_lin: m = new MachLin(f_idim,f_odim,bs); break;
    case file_header_mtype_sig: m = new MachSig(f_idim,f_odim,bs); break;
    case file_header_mtype_tanh: m = new MachTanh(f_idim,f_odim,bs); break;
    case file_header_mtype_softmax: m = new MachSoftmax(f_idim,f_odim,bs); break;
    case file_header_mtype_softmax_stable: m = new MachSoftmaxStable(f_idim,f_odim,bs); break;
    case file_header_mtype_lin_rectif: m = new MachLinRectif(f_idim,f_odim,bs); break;
    case file_header_mtype_multi: m = new MachMulti(); break;
    case file_header_mtype_mseq: m = new MachSeq(); break;
    //case file_header_mtype_mstack: m = new MachStack; break;
    case file_header_mtype_mpar: m = new MachPar(); break;
    case file_header_mtype_msplit1: m = new MachSplit1; break;
    case file_header_mtype_msplit: m = new MachSplit; break;
    default:
      ErrorN("unknown machine type in file (%d)", mtype);
  }
  if (!m) Error("no valid machine loaded");

    // read rest of (machine specific) params
  m->ReadParams(inpf);

  int s;
  inpf.read((char*) &s,sizeof(int));  // number of elements
  inpf.read((char*) &v,sizeof(int));  // size in bytes of each element
  if (v != sizeof(REAL)) {
    ErrorN( "binary data on file uses %d bytes while the current code is compiled for %lu bytes\n", v, sizeof(REAL));
  }
  m->ReadData(inpf, s, bs);
  // TODO: check EOF

  return m;
}

//-----------------------------------------------
// Tools
//-----------------------------------------------

void Mach::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << " - dimensions: in=" << idim << ", out=" << odim << endl;
    cout << " - number of parallel examples=" << bsize << endl;
    if (drop_out>0)
      cout << " - drop-out: " <<  drop_out << endl;
    cout << " - number of passes: " << nb_forw << "/" << nb_backw << endl;
  }
  else {
    if (drop_out>0)
      printf("%sMach %d-%d, bs=%d, drop-out=%4.2f, passes=%lu/%lu", txt, idim, odim, bsize, drop_out, nb_forw, nb_backw);
    else
      printf("%sMach %d-%d, bs=%d, passes=%lu/%lu", txt, idim, odim, bsize, nb_forw, nb_backw);
#ifdef BLAS_CUDA
    printf(", on GPU %d", cuda_dev);
#endif
    tm.disp(", ");
    printf("\n");
  }
}

//-----------------------------------------------
// Training
//-----------------------------------------------

void Mach::Forw(int eff_bsize)
{
  if (idim!=odim)
    Error("Mach::Forw(): call to default Forw() function with different dimensions");
  if (eff_bsize<=0) eff_bsize=bsize;
  if (!data_in)
    Error("Mach::Forw(): input data is not set");

  tm.start();

#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
  COPY(eff_bsize*idim,data_in,1,data_out,1); // this does work on host or GPU
#else
  int dim=eff_bsize*idim;
  COPY(&dim,data_in,&inc1,data_out,&inc1); // this does work on host or GPU
#endif
  nb_forw += (ulong) eff_bsize;

  tm.stop();
}

void Mach::Backw (const float lrate, const float wdecay, int eff_bsize)
{
  if (idim!=odim)
    Error("Mach::Backw(): call to default Train() function with different dimensions");
  if (!grad_out)
    Error("Mach::Backw(): output gradient is not set");

  if (eff_bsize<=0) eff_bsize=bsize;
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
  COPY(eff_bsize*idim,grad_out,1,grad_in,1);
#else
  memcpy(grad_in,grad_out,eff_bsize*idim*sizeof(REAL));
#endif
  nb_backw += (ulong) eff_bsize;
}

//******************************************

void GpuUnlock()
{
#ifdef BLAS_CUDA
  ofstream lfs;
  char lfname[LOCK_FNAME_LEN] = LOCK_FNAME;

    // removing all lock-files
  for (uint ii=0; ii<cuda_devs.size(); ii++) {
    sprintf(lfname,LOCK_FNAME,getpid(),cuda_devs[ii]);
    if (unlink(lfname)) {
      cerr << " - ERROR: removing lock file " << lfname << endl;
    }
  } 
#endif
}
