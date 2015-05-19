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
 * $Id: Mach.h,v 1.42 2014/03/25 21:52:53 schwenk Exp $
 */

#ifndef _Machine_h
#define _Machine_h

#include <iostream>
#include <fstream>
#include "Tools.h"
#include "Blas.h"
#include "Timer.h"

// list of all known machine types,
// this is needed for the general file read function

#define file_header_name "HPerf"
#define file_header_version1 1		// initial version
#define file_header_version2 2		// 2013/12/08: switched to ulong for nb_forw and nb_backw
#define file_header_version file_header_version2
#define file_header_size 16

#define file_header_mtype_base		0
#define file_header_mtype_tab		1
#define file_header_mtype_tabsh		2
#define file_header_mtype_lin		3
#define file_header_mtype_sig		4
#define file_header_mtype_tanh		5
#define file_header_mtype_softmax	6
#define file_header_mtype_stab		7
#define file_header_mtype_softmax_stable 8
#define file_header_mtype_lin_rectif	9
#define file_header_mtype_multi		16
#define file_header_mtype_mseq		17
#define file_header_mtype_msplit1	18
#define file_header_mtype_mpar		19
#define file_header_mtype_msplit	20
#define file_header_mtype_mjoin		21
#define file_header_mtype_combined	31
#define file_header_mtype_max		32
#define file_header_mtype_avr		33

class Mach
{
private:
  void do_alloc();	// perform allocation of dynamic data structures
protected:
  int idim, odim;		// input and output dimension
  int bsize;			// block size (nb of example used in parallel)
  ulong nb_forw;		// nb of forward examples processed
  ulong nb_backw;		// nb of backward examples processed
    // drop-out
  REAL drop_out;		// P for dropout, <=0: not used (default)
  REAL *drop_out_rand;		// random values for the whole output vector
   // CUDA: the following four variables refer to device memory
  REAL *data_in;		// input data (pointer)
				// CUDA: we need to allocate device memory
  REAL *data_out;		// output data (allocated by machine)
  REAL *grad_in;		// input gradients (allocated by machine)
  REAL *grad_out;		// output gradients (pointer)
				// CUDA: we need to allocate device memory
  Timer   tm;			// count real and user time
#ifdef BLAS_CUDA
  int	cuda_dev;  		// CUDA device; this is needed to run on multiple devices in parallel
#endif
  // File I/O, the following functions can be overloaded by subclass
  // the main functions Read() and Write() should not be modified !
  virtual void ReadParams(ifstream&, bool=true); // read all params
  virtual void ReadData(ifstream&, size_t, int=0); // read binary data
  virtual void WriteParams(ofstream&); // write all params
  virtual void WriteData(ofstream&); // write binary data
  Mach(const Mach &, const int=0);			// create a copy of the machine
public:
  Mach(const int=0, const int=0, const int=128, const ulong=0, const ulong=0);
  virtual ~Mach();
  virtual Mach *Clone() {return new Mach(*this);}	// create a copy of the machine
    // Tools
  virtual int GetMType() {return file_header_mtype_base;};	// get type of machine
  virtual int GetIdim() {return idim;}
  int GetOdim() {return odim;}
  int GetBsize() {return bsize;}
  virtual void SetBsize(int bs) {
    if (bs<1) Error("wrong value in SetBsize()"); else bsize=bs; }
  ulong GetNbForw() {return nb_forw;}
  ulong GetNbBackw() {return nb_backw;}
  virtual void ResetNbEx() {nb_forw=nb_backw=0;}
  virtual int GetNbParams() {return 0;}		// return the nbr of allocated parameters 
  virtual REAL* GetDataIn() {return data_in;}	// return pointer on input data for chaining
  virtual REAL* GetDataOut() {return data_out;}	// return pointer on output data for chaining
  virtual REAL* GetGradIn() {return grad_in;}	// return pointer on input gradient for chaining
  virtual REAL* GetGradOut() {return grad_out;} // return pointer on output gradient for chaining
  virtual void SetDataIn(REAL *data) {data_in=data;} // set pointer of input data
  virtual void SetGradOut(REAL *data) {grad_out=data;} // set pointer of output gradient 
  virtual void SetDropOut(const REAL v) {	// set drop-out fraction
    if (v>=1.0) Error("SetDropOut: the value must be smaller than 1.0");
    printf("MachLin::SetDropOut: %dx%d =%f\n",idim,odim,v);
    drop_out=v;
  }
#ifdef BLAS_CUDA
  int GetCudaDevice() { return cuda_dev; }	// return CUDA device used for this machine
#endif
  virtual void Info(bool=false, char *txt=(char*)" - ");// display (detailed) information on machine
    // FILE IO
  static Mach *Read(ifstream&, int=0);	// read class from a stream
  void Write(ofstream&); // write content of class to a stream
    // Training
  virtual void Forw(int=0);	// calculate outputs for current inputs
    // backprop gradients from output to input and update all weights
  virtual void Backw (const float lrate, const float wdecay, int =0);
};

void GpuUnlock();

#endif
