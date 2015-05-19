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
 * $Id: MachTab.h,v 1.21 2014/03/25 21:52:53 schwenk Exp $
 *
 * Table lookup machine:
 *   - input = index in table
 *   - output = ith line of table
 */

#ifndef _MachTab_h
#define _MachTab_h

#include <pthread.h>
#include "Mach.h"

class MachTab : public Mach
{
private:
  bool ext_table;	// flag to indicate whether table was provided at constructor with an external address
#ifdef BLAS_CUDA
  REAL *tmp_inp;	// temporary storage to get machine back to host
#endif
  virtual void do_alloc();	// perform allocation of dynamic data structures
protected:
  REAL *t;		// look-up table
  int *t_shared;	// number of objects sharing the look-up table
  pthread_mutex_t *t_mutex;	// mutex used to share look-up table
  virtual void WriteParams(ofstream&);
  virtual void ReadParams(ifstream&, bool =true);
  virtual void ReadData(ifstream&, size_t, int=0); // read binary data
  virtual void WriteData(ofstream&); // write binary data
  virtual int GetIdim() {return 1; } // we use idim internally as the dim of the table entries
  MachTab(const MachTab &);			// create a copy of the machine
public:
  MachTab(const int=1, const int=1, const int=128, const ulong=0, const ulong=0); // TODO: idim,odim init ??
  MachTab(REAL*, const int, const int, const int=128, const ulong=0, const ulong=0);	
  virtual ~MachTab();
  virtual MachTab *Clone() {return new MachTab(*this);}	// create a copy of the machine
  virtual int GetMType() {return file_header_mtype_tab;};	// get type of machine
  virtual int GetNbParams() {return ext_table ? 0 : idim*odim;}	// return the nbr of allocated parameters 
  virtual int GetMaxInpVal() {return idim;}		// all input values must be smaller than this, if not segfault ...
  virtual void TableConst(const REAL val);		// init table with constant values
  virtual void TableRandom(const REAL range);		// random init of table in [-range, range]
  virtual REAL *GetTabAdr() {return t; }		// 
  virtual void SetTabAdr(REAL *p_adr) {t=p_adr; }	// 
  virtual void Info(bool=false, char *txt=(char*)"");	// display (detailed) information on machine
  virtual void Forw(int=0);	// calculate outputs for current inputs
    // backprop gradients from output to input and update all weights
  virtual void Backw (const float lrate, const float wdecay, int=0);
};

#endif
