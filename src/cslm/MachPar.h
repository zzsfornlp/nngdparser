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
 * $Id: MachPar.h,v 1.15 2014/03/25 21:52:53 schwenk Exp $
 *
 *  Parallel machine:
 *   - put several machine in parallel with a concatenated input and output layer
 *   - the dimensions of the input and output layers may be different
 */

#ifndef _MachPar_h
#define _MachPar_h

using namespace std;
#include <vector>

#include "MachMulti.h"

class MachPar : public MachMulti
{
private:
  void do_alloc();	// perform allocation of dynamic data structures
protected:
  virtual void ReadData(ifstream&, size_t, int=0); // read binary data
  MachPar(const MachPar &);			// create a copy of the machine (without submachines)
public:
  MachPar();	// create initial sequence with no machine
  virtual ~MachPar();
  virtual MachPar *Clone();			// create a copy of the machine and all submachines
  virtual int GetMType() {return file_header_mtype_mpar;};	// get type of machine
    // redfine connecting functions
  virtual void SetDataIn(REAL*);	// set pointer of input data
  virtual void SetGradOut(REAL*);	// set pointer of output gradient 
    // add and remove machines
  virtual void MachAdd(Mach*); // add new machine after the existing ones
  virtual Mach *MachDel();
    // standard functions
  virtual void Info(bool=false, char *txt=(char*)"");	// display (detailed) information on machine
  virtual void Forw(int=0);	// calculate outputs for current inputs
  virtual void Backw(const float lrate, const float wdecay, int=0);	// calculate gradients at input for current gradients at output
};

#endif
