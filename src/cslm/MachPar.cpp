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
 * $Id: MachPar.cpp,v 1.34 2014/03/25 21:52:53 schwenk Exp $
 */

using namespace std;
#include <iostream>

#include "Tools.h"
#include "MachTab.h"
#include "MachPar.h"

void MachPar::do_alloc()
{
#ifdef BLAS_CUDA
  cudaSetDevice(cuda_dev);
  if (data_out) cublasFree(data_out);
  if (grad_in) cublasFree(grad_in);

  data_out = cuda_alloc(odim*bsize, "output data of parallel machine");
  grad_in = cuda_alloc(idim*bsize, "input gradient of parallel machine");

#else
  if (data_out) delete [] data_out;
  if (grad_in) delete [] grad_in;
  data_out = (odim*bsize>0) ? new REAL[odim*bsize] : NULL;
  grad_in = (idim*bsize>0) ? new REAL[idim*bsize] : NULL;
#endif
}


MachPar::MachPar()
 : MachMulti()
{
}

MachPar::MachPar(const MachPar &m)
 : MachMulti(m)
{
}

MachPar::~MachPar()
{
  // data_out and grad_in will be freed by Mach::~Mach()
}

MachPar *MachPar::Clone()
{
  MachPar *m = new MachPar(*this);
  if (m != NULL)
    m->CloneSubmachs(*this);
  return m;
}
 
void MachPar::MachAdd(Mach *new_mach)
{
  if (machs.empty()) {
    machs.push_back(new_mach);
	// think about freeing memory
    idim=new_mach->GetIdim();
    odim=new_mach->GetOdim();
    bsize=new_mach->GetBsize();
    data_in=NULL; // will be set by MachPar::SetDataIn()
    data_out=NULL;
    grad_in = NULL;
    grad_out = NULL;
    do_alloc();
    new_mach->SetGradOut(grad_out);
  }
  else {
    if (bsize!=new_mach->GetBsize())
      Error("bunch size of new parallel machine does not match");
    machs.push_back(new_mach);
 
      // resize input gradient and output data
    idim += new_mach->GetIdim();
    odim += new_mach->GetOdim();
    do_alloc();
  }
  activ_forw.push_back(true);
  activ_backw.push_back(true);
}

Mach *MachPar::MachDel()
{
  if (machs.empty()) {
    Error("impossible to delete element from parallel machine: is already empty");
  }
  
  Error("TODO");
  activ_forw.pop_back();
  activ_backw.pop_back();
  return NULL;
}

// set pointer of input data
void MachPar::SetDataIn(REAL *data)
{
  data_in=data;
    // set input data of indiv machines one after each other
    // this depends on the effective bsize !
  for (unsigned int m=0; m<machs.size(); m++) {
    machs[m]->SetDataIn(data);
    data += bsize*machs[m]->GetIdim();
  }
}

// set pointer of output gradient
void MachPar::SetGradOut(REAL *data)
{
  grad_out=data;
    // set output gradients of indiv machines one after each other
  for (unsigned int m=0; m<machs.size(); m++) {
    machs[m]->SetGradOut(data);
    data += bsize*machs[m]->GetOdim();
  }
}


//-----------------------------------------------
// File output
//-----------------------------------------------

void MachPar::ReadData(ifstream &inpf, size_t s, int bs)
{
  MachMulti::ReadData(inpf, s, bs);

     // calculate idim and odim and and allocate data_out and grad_in
  idim=odim=0;
  for (uint m=0; m<machs.size(); m++) {
    idim += machs[m]->GetIdim();
    odim += machs[m]->GetOdim();
  }
  bsize = machs[0]->GetBsize();
  do_alloc();

    // scanning for MachTab with shared addresses
  REAL *tadr=NULL;
  for (uint m=0; m<machs.size(); m++) {
    MachTab *mt= (MachTab*) machs[m];
    if (mt->GetMType()==file_header_mtype_tab) {
      if (mt->GetTabAdr()) {
        if (tadr) {
        }
        else {
        }
        tadr=mt->GetTabAdr();
      }
      else {
        mt->SetTabAdr(tadr);
      }
    }
  }
}

//
// Tools
//

void MachPar::Info(bool detailed, char *txt)
{
  if (detailed) {
    cout << "Information on parallel machine" << endl;
    MachMulti::Info(detailed);
  }
  else {
    printf("%sParallel machine %d-%d, bs=%d, passes=%lu/%lu", txt, idim, odim, bsize, nb_forw, nb_backw);
    tm.disp(", ");
    printf("\n");
    char ntxt[512];
    sprintf(ntxt,"%s  ", txt);
    for (unsigned int i=0; i<machs.size(); i++) machs[i]->Info(detailed, ntxt);
  }
  printf("%stotal number of parameters: %d (%d MBytes)\n", txt, GetNbParams(), (int) (GetNbParams()*sizeof(REAL)/1048576));
}

// TODO we do not organize correcty the input in the forward and backward pass with bunch mode !
// TODO since this is wrongly done at the input and output we finally get the corrrect result
// TODO but only when combine identical machines (like MachTab with shared codes)


// forward pass for all machines and copy output into cumulated output
void MachPar::Forw(int eff_bsize)
{
  if (machs.empty())
    Error("called Forw() for an empty parallel machine");

  debugMachInp("MachPar",data_in,idim,odim,eff_bsize);

  tm.start();
  if (eff_bsize<=0) eff_bsize=bsize;

      // we need to set the pointers to the input data of indiv machines
      // one after each other since this depends on the effective bsize !

  REAL *iptr=data_in;
  REAL *optr=data_out;
  for (unsigned int m=0; m<machs.size(); m++) {
    if (activ_forw[m]) {
      machs[m]->SetDataIn(iptr);
      machs[m]->Forw(eff_bsize);
#ifdef BLAS_CUDA
      cudaSetDevice(cuda_dev); // TODO: does this slow down ?
      nppsCopy_32f(machs[m]->GetDataOut(),optr,eff_bsize*machs[m]->GetOdim());
#else
      memcpy(optr, machs[m]->GetDataOut(), eff_bsize*machs[m]->GetOdim()*sizeof(REAL));
#endif
    }
    else {
    }
    iptr += eff_bsize*machs[m]->GetIdim();
    optr += eff_bsize*machs[m]->GetOdim();
  }
  nb_forw += eff_bsize; 

  tm.stop();
  debugMachOutp("MachPar",data_out,idim,odim,eff_bsize);
}

// backward pass for all machines and copy input gradient into cumulated gradient
void MachPar::Backw(const float lrate, const float wdecay, int eff_bsize)
{
  if (machs.empty())
    Error("called Backw() for an empty parallel machine");
  if (eff_bsize<=0) eff_bsize=bsize;
 
  tm.start();

      // we need to set the pointers to output gradients of indiv machines
      // one after each other since this depends on the effective bsize !

  REAL *gptr=grad_in;
  REAL *optr=grad_out;
  for (unsigned int m=0; m<machs.size(); m++) {
    if (activ_backw[m]) {
      machs[m]->SetGradOut(optr);
      machs[m]->Backw(lrate,wdecay,eff_bsize);
#ifdef BLAS_CUDA
      cudaSetDevice(cuda_dev); // TODO: does this slow down ?
      nppsCopy_32f(machs[m]->GetGradIn(),gptr,eff_bsize*machs[m]->GetIdim());
#else
      memcpy(gptr, machs[m]->GetGradIn(), eff_bsize*machs[m]->GetIdim()*sizeof(REAL));
#endif
    }
    else {
    }
    optr += eff_bsize*machs[m]->GetOdim();
    gptr += eff_bsize*machs[m]->GetIdim();
  }
  nb_backw += eff_bsize; 

  tm.stop();
}

