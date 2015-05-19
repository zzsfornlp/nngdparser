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
 * $Id: MachLin.h,v 1.28 2014/03/25 21:52:53 schwenk Exp $
 *
 * linear machine:  output = weights * input + biases
 */

#ifndef _MachLin_h
#define _MachLin_h

#include <pthread.h>
#include "Mach.h"

class MachLin : public Mach
{
	friend class CslmInterface;
protected:
  int  nb_params;	// number of params for max-out
   // CUDA: the following two variables refer to device memory
  REAL *b;		// biases
  REAL *w;		// weights, stored in BLAS format, e.g. COLUMN major !
  int *bw_shared;	// number of objects sharing biases and weights
  pthread_mutex_t *bw_mutex;	// mutex used to share biases and weights
  virtual void ReadData(ifstream&, size_t, int=0); // read binary data
  virtual void WriteData(ofstream&); // write binary data
  MachLin(const MachLin &);			// create a copy of the machine, sharing the parameters
public:
  MachLin(const int=0, const int=0, const int=128, const ulong=0, const ulong=0);	
  virtual ~MachLin();
  virtual MachLin *Clone() {return new MachLin(*this);}	// create a copy of the machine, sharing the parameters
  virtual int GetNbParams() {return idim*odim+odim;}		// return the nbr of allocated parameters 
  virtual int GetMType() {return file_header_mtype_lin;};	// get type of machine
  virtual void BiasConst(const REAL val);			// init biases with constant values
  virtual void BiasRandom(const REAL range);			// random init of biases in [-range, range]
  virtual void WeightsConst(const REAL val);			// init weights with constant values
  virtual void WeightsID(const REAL =1.0);			// init weights to identity transformation
  virtual void WeightsRandom(const REAL range);			// random init of weights in [-range, range]
  virtual void WeightsRandomFanI(const REAL range=sqrt(6.0));	// random init of weights in fct of fan-in
  virtual void WeightsRandomFanIO(const REAL range=sqrt(6.0));	// random init of weights in fct of fan-in and fan-out
  virtual void Info(bool=false, char *txt=(char*)"");		// display (detailed) information on machine
  virtual void Forw(int=0);	// calculate outputs for current inputs
    // backprop gradients from output to input and update all weights
  virtual void Backw (const float lrate, const float wdecay, int=0);
  virtual void Debug ();
};

#endif
