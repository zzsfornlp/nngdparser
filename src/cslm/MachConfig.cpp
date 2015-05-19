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
 * $Id: MachConfig.cpp,v 1.12 2014/03/25 21:52:53 schwenk Exp $
 */

#include <boost/program_options/errors.hpp>
#include <boost/program_options/parsers.hpp>
#include <cstring>
#include <strings.h>
#include "MachConfig.h"
#include "MachLinRectif.h"
#include "MachPar.h"
#include "MachSeq.h"
#include "MachSig.h"
#include "MachSoftmax.h"
#include "MachSoftmaxStable.h"
#include "MachSplit.h"
#include "MachSplit1.h"
//#include "MachStab.h"
//#include "MachStacked.h"
//#include "MachTabSh.h"
#include "MachTanh.h"
namespace bpo = boost::program_options;

/**
 * creates a machine configuration reader
 * @param bNeedConfFile true if configuration file is required on command line, false otherwise
 * @param rInitBias general value for random initialization of the bias (0.1 by default)
 */
MachConfig::MachConfig (bool bNeedConfFile, REAL rInitBias) :
      bSelectedOptions(false),
      bHelpRequest(false),
      bNeedConfFile(bNeedConfFile),
      bReadMachOnly(false),
      rInitBias(rInitBias),
      eErrorCode(MachConfig::NoError),
      odCommandLine("Command line options"),
      odSelectedConfig("Configuration options")
{
  /* set general options (in command line and configuration file) */

  // general options in command line only
  this->odCommandLine.add_options()
          ("help"                 , "produce help message")
          ("config-file,c"        , bpo::value< std::vector<std::string> >(), "configuration file (can be set without option name)")
          ;
  this->podCommandLine.add("config-file", -1); // command line may contain configuration file name without option name

  // general options in configuration file and selectable for command line
  this->odGeneralConfig.add_options()
          ("mach,m"               , opt_sem<std::string>::new_sem(), "file name of the machine")
          ("src-word-list,s"      , opt_sem<std::string>::new_sem(), "word list of the source vocabulary")
          ("tgt-word-list,w"      , opt_sem<std::string>::new_sem(), "word list of the vocabulary and counts (used to select the most frequent words)")
          ("word-list,w"          , opt_sem<std::string>::new_sem(), "word list of the vocabulary and counts (used to select the most frequent words)")
          ("input-file,i"         , opt_sem<std::string>::new_sem(), "file name of the input n-best list")
          ("output-file,o"        , opt_sem<std::string>::new_sem(), "file name of the output n-best list")
          ("source-file,S"        , opt_sem<std::string>::new_sem(), "file name of the file with source sentences (needed for TM rescoring)")
          ("phrase-table"         , opt_sem<std::string>::new_sem(), "rescore with a Moses on-disk phrase table")
          ("phrase-table2"        , opt_sem<std::string>::new_sem(), "use a secondary Moses phrase table")
          ("test-data,t"          , opt_sem<std::string>::new_sem(), "test data")
          ("train-data,t"         , opt_sem<std::string>::new_sem(), "training data")
          ("dev-data,d"           , opt_sem<std::string>::new_sem(), "development data (optional)")
          ("lm,l"                 , opt_sem<std::string>::new_sem(), "file name of the machine (only necessary when using short lists)")
          ("output-probas"        , opt_sem<std::string>::new_sem(), "write sequence of log-probas to file (optional)")
          ("cslm,c"               , opt_sem<std::string>::new_sem(), "rescore with a CSLM")
          ("vocab,v"              , opt_sem<std::string>::new_sem(), "word-list to be used with the CSLM")
          ("cstm,C"               , opt_sem<std::string>::new_sem(), "rescore with a CSTM")
          ("vocab-source,b"       , opt_sem<std::string>::new_sem(), "source word-list to be used with the CSTM")
          ("vocab-target,B"       , opt_sem<std::string>::new_sem(), "target word-list to be used with the CSTM")
          ("weights,w"            , opt_sem<std::string>::new_sem(), "coefficients of the feature functions")
          ("tm-scores,N"          , opt_sem<std::string>::new_sem()->default_value("4:0"), "specification of the TM scores to be used (default first 4)")
          ("inn,I"                , opt_sem<int> ::new_sem(                     )->default_value(    0  ), "number of hypothesis to read per n-best (default all)")
          ("outn,O"               , opt_sem<int> ::new_sem(                     )->default_value(    0  ), "number of hypothesis to write per n-best (default all)")
          ("offs,a"               , opt_sem<int> ::new_sem(                     )->default_value(    0  ), "add offset to n-best ID (useful for separately generated n-bests)")
          ("num-scores,n"         , opt_sem<int> ::new_sem(                     )->default_value(    5  ), "number of scores in phrase table")
          ("ctxt-in,c"            , opt_sem<int> ::new_sem(                     )->default_value(    7  ), "input context size")
          ("ctxt-out,C"           , opt_sem<int> ::new_sem(                     )->default_value(    7  ), "output context size")
          ("curr-iter,C"          , opt_sem<int> ::new_sem(                     )->default_value(    0  ), "current iteration when continuing training of a neural network")
          ("last-iter,I"          , opt_sem<int> ::new_sem(                     )->default_value(   10  ), "last iteration of neural network")
          ("order"                , opt_sem<int> ::new_sem(                     )->default_value(    4  ), "order of the LM to apply on the test data (must match CSLM, but not necessarily back-off LM)")
          ("mode,M"               , opt_sem<int> ::new_sem(                     )->default_value(    3  ), "mode of the data (1=IGN_BOS 2=IGN_UNK 4=IGN_UNK_ALL, 8=IGN_EOS)")
          ("lm-pos,p"             , opt_sem<int> ::new_sem(                     )->default_value(    0  ), "position of LM score (1..n, 0 means to append it)")
          ("tm-pos,P"             , opt_sem<int> ::new_sem(                     )->default_value(    0  ), "position of the TM scores, up to 4 values")
          ("buf-size,b"           , opt_sem<int> ::new_sem(                     )->default_value(16384  ), "buffer size")
          ("block-size,B"         , opt_sem<int> ::new_sem(&this->iBlockSize    )->default_value(  128  ), "block size for faster training")
          ("drop-out,O"           , opt_sem<REAL>::new_sem(&this->rPercDropOut  )->default_value(   -1.0), "percentage of neurons to be used for drop-out [0-1] (set by default to -1 to turn it off)")
          ("random-init-project,r", opt_sem<REAL>::new_sem(&this->rInitProjLayer)->default_value(    0.1), "value for random initialization of the projection layer")
          ("random-init-weights,R", opt_sem<REAL>::new_sem(&this->rInitWeights  )->default_value(    0.1), "value for random initialization of the weights")
          ("lrate-beg,L"          , opt_sem<REAL>::new_sem(                     )->default_value(  5E-03), "initial learning rate")
          ("lrate-mult,M"         , opt_sem<REAL>::new_sem(                     )->default_value(  7E-08), "learning rate multiplier for exponential decrease")
          ("weight-decay,W"       , opt_sem<REAL>::new_sem(                     )->default_value(  3E-05), "coefficient of weight decay")
          ("backward-tm,V"        , opt_sem<bool>::new_sem()->zero_tokens(), "use an inverse back-ward translation model")
          ("renormal,R"           , opt_sem<bool>::new_sem()->zero_tokens(), "renormalize all probabilities, slow for large short-lists")
          ("recalc,r"             , opt_sem<bool>::new_sem()->zero_tokens(), "recalculate global scores")
          ("sort,s"               , opt_sem<bool>::new_sem()->zero_tokens(), "sort n-best list according to the global scores")
          ("lexical,h"            , opt_sem<bool>::new_sem()->zero_tokens(), "report number of lexically different hypothesis")
          ("server,X"             , opt_sem<bool>::new_sem()->zero_tokens(), "run in server mode listening to a named pipe to get weights for new solution extraction")
#ifdef BLAS_CUDA
          ("cuda-device,D"        , opt_sem<std::vector<int> >::new_sem()->default_value(std::vector<int>(1,0),"0"), "select CUDA device")
#endif
          ;


  /* set machine names */

  // machine names are defined in configuration file options to be recognized as valid options
  this->odMachineTypes.add_options()
          ("machine.Mach"         , bpo::value<std::vector<std::string> >())
          ("machine.Tab"          , bpo::value<std::vector<std::string> >())
          ("machine.Linear"       , bpo::value<std::vector<std::string> >())
          ("machine.LinRectif"    , bpo::value<std::vector<std::string> >())
          ("machine.Sig"          , bpo::value<std::vector<std::string> >())
          ("machine.Tanh"         , bpo::value<std::vector<std::string> >())
          ("machine.Softmax"      , bpo::value<std::vector<std::string> >())
          ("machine.SoftmaxStable", bpo::value<std::vector<std::string> >())
          ("machine.Multi"        , bpo::value<std::vector<std::string> >())
          ("machine.Sequential"   , bpo::value<std::vector<std::string> >())
          ("machine.Parallel"     , bpo::value<std::vector<std::string> >())
          ("machine.Split"        , bpo::value<std::vector<std::string> >())
          ("machine.Split1"       , bpo::value<std::vector<std::string> >())
          ("machine.Join"         , bpo::value<std::vector<std::string> >())
          ;
  this->odGeneralConfig.add(this->odMachineTypes);


  /* set machine specific options */

  // machine options for many machine types except multiple machines
  this->odMachineConf.add_options()
          ("input-dim"            , bpo::value<int> ()->required(), "input dimension")
          ("output-dim"           , bpo::value<int> ()->required(), "output dimension")
          ("nb-forward"           , bpo::value<int> ()->default_value(0), "forward number")
          ("nb-backward"          , bpo::value<int> ()->default_value(0), "backward number")
          ;

  // machine options for all machine types (including multiple machines)
  this->odMachMultiConf.add_options()
          ("drop-out"             , bpo::value<REAL>(), "percentage of neurons to be used for drop-out [0-1], set to -1 to turn it off")
          ("block-size"           , bpo::value<int> (), "block size for faster training")
          ("init-from-file"       , bpo::value<std::string>(), "name of file containing all machine data")
          ;
  this->odMachineConf.add(this->odMachMultiConf);

  // machine options for linear machines (base class MachLin)
  this->odMachLinConf.add_options()
          ("const-init-weights"   , bpo::value<REAL>(), "constant value for initialization of the weights")
          ("ident-init-weights"   , bpo::value<REAL>(), "initialization of the weights by identity transformation")
          ("fani-init-weights"    , bpo::value<REAL>(), "random initialization of the weights by function of fan-in")
          ("fanio-init-weights"   , bpo::value<REAL>(), "random initialization of the weights by function of fan-in and fan-out")
          ("random-init-weights"  , bpo::value<REAL>(), "value for random initialization of the weights (method used by default with general value)")
          ("const-init-bias"      , bpo::value<REAL>(), "constant value for initialization of the bias")
          ("random-init-bias"     , bpo::value<REAL>(), "value for random initialization of the bias (method used by default with general value)")
          ;
  this->odMachLinConf.add(this->odMachineConf);

  // machine options for table lookup machines (base class MachTab)
  this->odMachTabConf.add_options()
          ("const-init-project"   , bpo::value<REAL>(), "constant value for initialization of the projection layer")
          ("random-init-project"  , bpo::value<REAL>(), "value for random initialization of the projection layer (method used by default with general value)")
          ;
  this->odMachTabConf.add(this->odMachineConf);
}

/**
 * parses options from command line and configuration file
 * @param iArgCount number of command line arguments
 * @param sArgTable table of command line arguments
 * @return false in case of error or help request, true otherwise
 * @note error code is set if an error occurred
 */
bool MachConfig::parse_options (int iArgCount, char *sArgTable[])
{
  this->vmGeneralOptions.clear();

  // program name
  if (iArgCount > 0) {
    this->sProgName = sArgTable[0];
    size_t stEndPath = this->sProgName.find_last_of("/\\");
    if (stEndPath != std::string::npos)
      this->sProgName.erase(0, stEndPath + 1);
  }
  else
    this->sProgName.clear();

  // set option list used by the application
  bpo::options_description odUsedOptions;
  odUsedOptions.add(this->odCommandLine);
  odUsedOptions.add(this->odSelectedConfig);

  // parse command line
  try {
    bpo::store(bpo::command_line_parser(iArgCount, sArgTable).options(odUsedOptions).positional(this->podCommandLine).run(), this->vmGeneralOptions);

    // verify help option
    this->bHelpRequest = (this->vmGeneralOptions.count("help") > 0);
    if (this->bHelpRequest)
      return false;

    // get configuration file name
    std::vector<std::string> vs;
    std::string sConfFileOpt("config-file");
    if (this->vmGeneralOptions.count(sConfFileOpt) > 0)
      vs = this->vmGeneralOptions[sConfFileOpt].as< std::vector<std::string> >();
    switch (vs.size()) {
    case 1:
      this->sConfFile = vs.front();
      break;
    case 0:
      this->sConfFile.clear();
      if (this->bNeedConfFile) {
        // error: configuration file is required
        throw bpo::required_option(sConfFileOpt);
      }
      else {
        // don't parse configuration file, so notify command line parsing
        bpo::notify(this->vmGeneralOptions);
        return true;
      }
      break;
    default:
      bpo::multiple_occurrences mo;
      mo.set_option_name(sConfFileOpt);
      throw mo;
      break;
    }

  } catch (bpo::error &e) {
    // error handling
    this->eErrorCode = MachConfig::CmdLineParsingError;
    this->ossErrorInfo.str(e.what());
    return false;
  }

  // open configuration file
  if (!this->open_file())
    return false;

  try {
    // parse configuration file and parse command line one more time (to be sure to use selected options with the good attributes)
    bpo::store(bpo::parse_config_file(this->ifsConf, this->odGeneralConfig), this->vmGeneralOptions);
    bpo::store(bpo::command_line_parser(iArgCount, sArgTable).options(odUsedOptions).positional(this->podCommandLine).run(), this->vmGeneralOptions);
    bpo::notify(this->vmGeneralOptions);
  } catch (bpo::error &e) {
    // error handling
    this->eErrorCode = MachConfig::ConfigParsingError;
    this->ossErrorInfo.str(e.what());
    return false;
  }

  // remove unused information (machine structure which will be read without boost)
  const std::vector<boost::shared_ptr<bpo::option_description> >& vodMachOpt = this->odMachineTypes.options();
  std::vector<boost::shared_ptr<bpo::option_description> >::const_iterator iEnd = vodMachOpt.end();
  for (std::vector<boost::shared_ptr<bpo::option_description> >::const_iterator iO = vodMachOpt.begin() ; iO != iEnd ; iO++) {
    bpo::option_description *pod = iO->get();
    if (pod != NULL)
      this->vmGeneralOptions.erase(pod->long_name());
  }

  return true;
}

/**
 * prints help message on standard output
 */
void MachConfig::print_help () const
{
  std::cout <<
      "Usage: " << this->sProgName << " [options]" << std::endl <<
      "       " << this->sProgName << " configuration_file_name [options]" << std::endl <<
      std::endl << this->odCommandLine << std::endl;
  if (this->bSelectedOptions)
    std::cout << this->odSelectedConfig << std::endl;
}

/**
 * reads machine structure from configuration file
 * @return new machine object, or NULL in case of error
 * @note error code is set if an error occurred
 */
Mach *MachConfig::get_machine ()
{
  // open configuration file
  if (!this->open_file())
    return NULL;

  // search for "machine" group
  std::string sRead;
  char sMachGroup[] = "[machine]";
  do {
    this->ifsConf >> sRead;
    std::ios_base::iostate iost = this->ifsConf.rdstate();
    if (iost) {
      // error handling
      if (iost & std::ios_base::eofbit)
        this->eErrorCode = MachConfig::NoMachineGroup;
      else
        this->eErrorCode = MachConfig::ProbSearchMachGroup;
      return NULL;
    }
  } while (sRead != sMachGroup);

  // read machine structure
  this->bReadMachOnly = false;
  this->eErrorCode = MachConfig::NoError;
  Mach *pNextMach = NULL;
  this->read_next_machine(pNextMach, this->iBlockSize);
  if ((this->eErrorCode != MachConfig::NoError) && (pNextMach != NULL)) {
    delete pNextMach;
    return NULL;
  }
  else
    return pNextMach;
}

/**
 * get last error
 * @return error string
 */
std::string MachConfig::get_error_string() const
{
  std::string sError;

  // get string
  switch (this->eErrorCode) {
  case MachConfig::NoError:
    return std::string();
    break;
  case MachConfig::CmdLineParsingError:
    sError = "command line error: ";
    sError += this->ossErrorInfo.str();
    return sError;
    break;
  case MachConfig::ProbOpenConfigFile:
    sError = "can't open configuration file \"";
    sError += this->sConfFile;
    sError += '\"';
    return sError;
    break;
  case MachConfig::ConfigParsingError:
    sError = "configuration error: ";
    sError += this->ossErrorInfo.str();
    return sError;
    break;
  case MachConfig::NoMachineGroup:
    return "no [machine] group in configuration file";
    break;
  case MachConfig::ProbSearchMachGroup:
    return "internal error while searching [machine] group";
    break;
  case MachConfig::MachDescrIncomplete:
    return "machine description is not complete";
    break;
  case MachConfig::ProbReadMachName:
    return "internal error while reading machine type name";
    break;
  case MachConfig::UnknownMachType:
    sError = "unknown machine type \"";
    break;
  case MachConfig::UnknownMachCode:
    sError = "unknown machine code ";
    sError += this->ossErrorInfo.str();
    return sError;
    break;
  case MachConfig::MachWithoutEqualChar:
    sError = "no equal character after machine name in \"";
    break;
  case MachConfig::ProbReadMachParams:
    sError = "internal error while reading machine parameters in \"";
    break;
  case MachConfig::MachParamsParsingError:
    sError = "machine parameters error in \"";
    sError += this->ossErrorInfo.str();
    return sError;
    break;
  case MachConfig::ProbOpenMachineFile:
    sError = "can't open machine data file \"";
    break;
  case MachConfig::ProbAllocMachine:
    sError = "can't allocate machine \"";
    break;
  default:
    std::ostringstream oss;
    oss << "unknown error " << this->eErrorCode;
    return oss.str();
    break;
  };

  // append machine type
  sError += this->ossErrorInfo.str();
  sError += '\"';

  return sError;
}

/**
 * get file name of the machine (or void string if not set)
 * @note if mach option value is "%CONF", file name will be same as configuration file (without extension ".conf") followed by extension ".mach"
 */
std::string MachConfig::get_mach () const
{
  const boost::program_options::variable_value &vvM = this->vmGeneralOptions["mach"];
  if (vvM.empty())
    // mach option not set
    return std::string();
  else {
    const std::string &sMachOpt = vvM.as<std::string>();
    if ((sMachOpt == "%CONF") && !this->sConfFile.empty()) {
      size_t stConfFileLen = this->sConfFile.length();

      std::string sConfExt(".conf");
      size_t stConfExtLen = sConfExt.length();

      // verify config-file extension
      if (    (   stConfFileLen   >=  stConfExtLen    )
          &&  (this->sConfFile.compare(stConfFileLen - stConfExtLen, stConfExtLen, sConfExt) == 0)    )
        stConfFileLen -= stConfExtLen;

      // return mach value as config-file value with new extension
      std::string sMachVal(this->sConfFile, 0, stConfFileLen);
      sMachVal.append(".mach");
      return sMachVal;
    }
    else
      // return mach value as set
      return sMachOpt;
  }
}

/**
 * open configuration file
 * @return false in case of error, true otherwise
 */
bool MachConfig::open_file ()
{
  this->ifsConf.close();
  this->ifsConf.clear();

  this->ifsConf.open(this->sConfFile.c_str(), std::ios_base::in);
  if (this->ifsConf.fail()) {
    this->eErrorCode = MachConfig::ProbOpenConfigFile;
    return false;
  }
  else {
    this->ifsConf.clear();
    return true;
  }
}

/**
 * reads next machine block from configuration file
 * @param pNewMach set to new machine object pointer, or NULL if 'end' mark is read (and possibly in case of error)
 * @param iBlockSize block size for faster training
 * @param prLookUpTable look-up table for each MachTab in a MachPar (default NULL)
 * @return true if 'end' mark is read, false otherwise
 * @note error code is set if an error occurred
 */
bool MachConfig::read_next_machine (Mach *&pNewMach, int iBlockSize, REAL *prLookUpTable)
{
  // read machine type name
  std::string sMachType;
  this->ossErrorInfo.str(sMachType);
  this->ifsConf >> sMachType;
  std::ios_base::iostate iost = this->ifsConf.rdstate();
  if (iost) {
    // error handling
    if (iost & std::ios_base::eofbit)
      this->eErrorCode = MachConfig::MachDescrIncomplete;
    else
      this->eErrorCode = MachConfig::ProbReadMachName;
    this->ossErrorInfo << sMachType;
    pNewMach = NULL;
    return false;
  }

  // verify if name contains equal sign
  size_t stEqualPos = sMachType.find('=', 1);
  if (stEqualPos != std::string::npos) {
    this->ifsConf.seekg(stEqualPos - sMachType.length(), std::ios_base::cur);
    this->ifsConf.clear();
    sMachType.resize(stEqualPos);
  }
  this->ossErrorInfo << sMachType;

  // get machine type
  int iMachType;
  bool bMachLin   = false;
  bool bMachMulti = false;
  bool bMachTab   = false;
  const char *sMachType_cstr = sMachType.c_str();
  if (strcasecmp(sMachType_cstr, "#End") == 0) {
    pNewMach = NULL;
    return true;
  }
  else if (strcasecmp(sMachType_cstr, "Mach") == 0) {
    iMachType = file_header_mtype_base;
  }
  else if (strcasecmp(sMachType_cstr, "Tab") == 0) {
    iMachType = file_header_mtype_tab;
    bMachTab = true;
  }
  /*else if (strcasecmp(sMachType_cstr, "Tabsh") == 0) {
    iMachType = file_header_mtype_tabsh;
    bMachTab = true;
  }*/
  else if (strcasecmp(sMachType_cstr, "Linear") == 0) {
    iMachType = file_header_mtype_lin;
    bMachLin = true;
  }
  else if (strcasecmp(sMachType_cstr, "Sig") == 0) {
    iMachType = file_header_mtype_sig;
    bMachLin = true;
  }
  else if (strcasecmp(sMachType_cstr, "Tanh") == 0) {
    iMachType = file_header_mtype_tanh;
    bMachLin = true;
  }
  else if (strcasecmp(sMachType_cstr, "Softmax") == 0) {
    iMachType = file_header_mtype_softmax;
    bMachLin = true;
  }
  /*else if (strcasecmp(sMachType_cstr, "Stab") == 0) {
    iMachType = file_header_mtype_stab;
    bMachLin = true;
  }*/
  else if (strcasecmp(sMachType_cstr, "SoftmaxStable") == 0) {
    iMachType = file_header_mtype_softmax_stable;
    bMachLin = true;
  }
  else if (strcasecmp(sMachType_cstr, "LinRectif") == 0) {
    iMachType = file_header_mtype_lin_rectif;
    bMachLin = true;
  }
  else {
    bMachMulti = true;
    if (strcasecmp(sMachType_cstr, "Multi") == 0)
      iMachType = file_header_mtype_multi;
    else if (strcasecmp(sMachType_cstr, "Sequential") == 0)
      iMachType = file_header_mtype_mseq;
    else if (strcasecmp(sMachType_cstr, "Split1") == 0)
      iMachType = file_header_mtype_msplit1;
    else if (strcasecmp(sMachType_cstr, "Parallel") == 0)
      iMachType = file_header_mtype_mpar;
    else if (strcasecmp(sMachType_cstr, "Split") == 0)
      iMachType = file_header_mtype_msplit;
    else {
      // error handling
      this->eErrorCode = MachConfig::UnknownMachType;
      pNewMach = NULL;
      return false;
    }
  }

  // create machine
  if (bMachMulti)
    pNewMach = this->read_multi_machine (iMachType, iBlockSize);
  else
    pNewMach = this->read_simple_machine(iMachType, iBlockSize, prLookUpTable, bMachLin, bMachTab);
  return false;
}

/**
 * creates a multiple machine, reads his parameters and reads submachine blocks
 * @param iMachType type of multiple machine
 * @param iBlockSize block size for faster training
 * @return new machine object (may be NULL in case of error)
 * @note error code is set if an error occurred
 */
Mach *MachConfig::read_multi_machine (int iMachType, int iBlockSize)
{
  Mach *pNewMach = NULL;
  MachMulti *pMachMulti = NULL;
  bool bInitFromFile = false;

  // read machine parameters
  bpo::variables_map vmMachParams;
  if (!this->read_machine_parameters(this->odMachMultiConf, vmMachParams))
    return NULL;

  // get current block size (get current machine block size if defined, or block size in parameter)
  const boost::program_options::variable_value &vvBS = vmMachParams["block-size"];
  int iCurBlockSize = (vvBS.empty() ? iBlockSize : vvBS.as<int>());

  // verify if machine structure must be read without creating new object
  if (!this->bReadMachOnly) {

    // verify if machine data file name is available
    const boost::program_options::variable_value &vvIFF = vmMachParams["init-from-file"];
    bInitFromFile = !vvIFF.empty();

    // create new machine object
    if (bInitFromFile) {
      // read machine data from file
      pNewMach = this->read_machine_from_file(vvIFF.as<std::string>(), iCurBlockSize, vmMachParams);
      if (pNewMach == NULL)
        // error handling
        return NULL;
      this->bReadMachOnly = true;
    }
    else {
      // instantiate multi machine corresponding to given type
      switch (iMachType) {
      case file_header_mtype_multi:
        pMachMulti = new MachMulti;
        break;
      case file_header_mtype_mseq:
        pMachMulti = new MachSeq;
        break;
      case file_header_mtype_msplit1:
        pMachMulti = new MachSplit1;
        break;
      case file_header_mtype_mpar:
        pMachMulti = new MachPar;
        break;
      case file_header_mtype_msplit:
        pMachMulti = new MachSplit;
        break;
      default:
        this->eErrorCode = MachConfig::UnknownMachCode;
        this->ossErrorInfo.str(std::string());
        this->ossErrorInfo << iMachType;
        return NULL;
        break;
      }
      if (pMachMulti == NULL) {
        // error handling
        this->eErrorCode = MachConfig::ProbAllocMachine;
        return NULL;
      }
      pNewMach = pMachMulti;

      // apply drop-out parameter (current machine drop-out value if defined, or general value)
      const boost::program_options::variable_value &vvDO = vmMachParams["drop-out"];
      pNewMach->SetDropOut(vvDO.empty() ? this->rPercDropOut : vvDO.as<REAL>());
    }
  }

  // read submachines
#ifdef BLAS_CUDA
  int iMachDev = ((pMachMulti != NULL) ? pMachMulti->GetCudaDevice() : 0);
  size_t stDevIndex = 0;
  bool bChangeDev = ((cuda_devs.size() > 1) && (pMachMulti != NULL) && (
                          (iMachType == file_header_mtype_msplit)
                      ));
#endif
  REAL *prLookUpTable = NULL;
  do {
#ifdef BLAS_CUDA
    if (bChangeDev) {
      cudaSetDevice(cuda_devs[stDevIndex % cuda_devs.size()]);
      stDevIndex++;
    }
#endif
    Mach *pSubMach = NULL;
    if (this->read_next_machine(pSubMach, iCurBlockSize, prLookUpTable))
      break;
    else if (pSubMach != NULL) {
      // handle errors
      if (this->eErrorCode != MachConfig::NoError) {
        delete pSubMach;
        break;
      }

      // get look-up table of first MachTab in a MachPar
      int iSubMType = pSubMach->GetMType();
      if (   (prLookUpTable == NULL) && (iMachType == file_header_mtype_mpar)
          && ((iSubMType == file_header_mtype_tab) || (iSubMType == file_header_mtype_tabsh)) ) {
        prLookUpTable = static_cast<MachTab*>(pSubMach)->GetTabAdr();
      }

      // add new submachine to multi machine
      if (pMachMulti != NULL)
        pMachMulti->MachAdd(pSubMach);
    }
  } while (this->eErrorCode == MachConfig::NoError);
#ifdef BLAS_CUDA
  if (bChangeDev) cudaSetDevice(iMachDev); // reset to multi machine GPU
#endif

  if (bInitFromFile)
    this->bReadMachOnly = false;
  return pNewMach;
}

/**
 * creates a simple machine and reads his parameters
 * @param iMachType type of simple machine
 * @param iBlockSize block size for faster training
 * @param prLookUpTable look-up table for each MachTab in a MachPar (default NULL)
 * @param bMachLin true if the machine is a linear machine, default false otherwise
 * @param bMachTab true if the machine is a table lookup machine, default false otherwise
 * @return new machine object (may be NULL in case of error)
 * @note error code is set if an error occurred
 */
Mach *MachConfig::read_simple_machine (int iMachType, int iBlockSize, REAL *prLookUpTable, bool bMachLin, bool bMachTab)
{
  Mach *pNewMach = NULL;

  // read machine parameters
  bpo::variables_map vmMachParams;
  if (!this->read_machine_parameters (bMachLin ? this->odMachLinConf : (bMachTab ? this->odMachTabConf : this->odMachineConf), vmMachParams))
    return NULL;

  // verify if machine structure must be read without creating new object
  if (this->bReadMachOnly)
    return NULL;

  // get current block size (get current machine block size if defined, or block size in parameter)
  const boost::program_options::variable_value &vvBS = vmMachParams["block-size"];
  int iCurBlockSize = (vvBS.empty() ? iBlockSize : vvBS.as<int>());

  // create new machine object
  const boost::program_options::variable_value &vvIFF = vmMachParams["init-from-file"];
  if (!vvIFF.empty()) {
    // read machine data from file
    pNewMach = this->read_machine_from_file(vvIFF.as<std::string>(), iCurBlockSize, vmMachParams);
  }
  else {
    // get dimension values
    int iInputDim  = vmMachParams[ "input-dim"].as<int>();
    int iOutputDim = vmMachParams["output-dim"].as<int>();

    // get forward and backward numbers
    int iNbForward  = vmMachParams["nb-forward" ].as<int>();
    int iNbBackward = vmMachParams["nb-backward"].as<int>();

    // instantiate simple machine corresponding to given type
    MachLin *pMachLin = NULL;
    MachTab *pMachTab = NULL;
    switch (iMachType) {
    case file_header_mtype_base:
      pNewMach = new Mach(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
      break;
    case file_header_mtype_tab:
      if (prLookUpTable == NULL) {
        // first MachTab in a MachPar or not in a MachPar
        pNewMach = pMachTab = new MachTab(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
      }
      else {
        // other MachTab in a MachPar
        pNewMach = pMachTab = new MachTab(prLookUpTable, iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
      }
      break;
    /*case file_header_mtype_tabsh:
      if (prLookUpTable == NULL) {
          // first MachTab in a MachPar or not in a MachPar
        pNewMach = pMachTab = new MachTabSh(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
      }
      else {
          // other MachTab in a MachPar
        pNewMach = pMachTab = new MachTabSh(prLookUpTable, iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
      }
      break;*/
    case file_header_mtype_lin:
      pNewMach = pMachLin = new MachLin(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
      break;
    case file_header_mtype_lin_rectif:
      pNewMach = pMachLin = new MachLinRectif(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
      break;
    case file_header_mtype_sig:
      pNewMach = pMachLin = new MachSig(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
      break;
    case file_header_mtype_tanh:
      pNewMach = pMachLin = new MachTanh(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
      break;
    case file_header_mtype_softmax:
      pNewMach = pMachLin = new MachSoftmax(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
      break;
    /*case file_header_mtype_stab:
      pNewMach = pMachLin = MachStab(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
      break;*/
    case file_header_mtype_softmax_stable:
      pNewMach = pMachLin = new MachSoftmaxStable(iInputDim, iOutputDim, iCurBlockSize, iNbForward, iNbBackward);
      break;
    default:
      this->eErrorCode = MachConfig::UnknownMachCode;
      this->ossErrorInfo.str(std::string());
      this->ossErrorInfo << iMachType;
      return NULL;
      break;
    }
    if (pNewMach == NULL) {
      // error handling
      this->eErrorCode = MachConfig::ProbAllocMachine;
      return NULL;
    }

    // apply drop-out parameter (current machine drop-out value if defined, or general value)
    const boost::program_options::variable_value &vvDO = vmMachParams["drop-out"];
    pNewMach->SetDropOut(vvDO.empty() ? this->rPercDropOut : vvDO.as<REAL>());

    // initialize MachLin
    if (pMachLin != NULL)
      this->apply_machine_parameters(pMachLin, vmMachParams, true);

    // initialize MachTab
    if (pMachTab != NULL)
      this->apply_machine_parameters(pMachTab, vmMachParams, prLookUpTable == NULL);
  }

  return pNewMach;
}

/**
 * reads machine parameters and fills it in given map
 * @param odMachineConf available options for the machine
 * @param vmMachParams map filled with parameters read
 * @return false in case of error, true otherwise
 */
bool MachConfig::read_machine_parameters (const bpo::options_description &odMachineConf, bpo::variables_map &vmMachParams)
{
  // read equal character
  char cEqual = ' ';
  this->ifsConf >> cEqual;
  bool bNoEqualChar = (cEqual != '=');

  // read until end of line
  std::stringbuf sbParamsLine;
  this->ifsConf.get(sbParamsLine);
  std::string sParams = sbParamsLine.str();

  // handle errors
  if (this->ifsConf.bad() || bNoEqualChar) {
    if (bNoEqualChar)
      this->eErrorCode = MachConfig::MachWithoutEqualChar;
    else
      this->eErrorCode = MachConfig::ProbReadMachParams;
    this->ossErrorInfo << ' ' << cEqual << sParams;
    return false;
  }
  this->ifsConf.clear();

  // read machine parameters
  try {
    bpo::store(
        bpo::command_line_parser(std::vector<std::string>(1, sParams)).
        extra_style_parser(MachConfig::parse_mach_params).options(odMachineConf).run(), vmMachParams);
    bpo::notify(vmMachParams);
  }
  catch (bpo::error &e) {
    // error handling
    this->eErrorCode = MachConfig::MachParamsParsingError;
    this->ossErrorInfo << " =" << sParams << "\": " << e.what();
    return false;
  }

  return true;
}

/**
 * parses machine parameters (dimensions and other options)
 * @param vsTokens vector of tokens
 * @return vector of options
 * @note throws exception of class boost::program_options::error in case of error
 */
std::vector<bpo::option> MachConfig::parse_mach_params (const std::vector<std::string> &vsTokens)
{
  std::vector<bpo::option> voParsed;

  // put tokens in stream
  std::stringstream ssTokens;
  std::vector<std::string>::const_iterator iEnd = vsTokens.end();
  for (std::vector<std::string>::const_iterator iT = vsTokens.begin() ; iT != iEnd ; iT++)
    ssTokens << *iT << ' ';

  // read abbreviated dimension values
  int iInputDim = -1, iOutputDim = -1;
  char cCross = ' ';
  ssTokens >> iInputDim >> cCross >> iOutputDim; // ex: " 128 x 256 ", " 128 X 256 "
  if ((ssTokens.rdstate() == 0) && ((cCross == 'x') || (cCross == 'X'))) {
    // dimensions available
    std::ostringstream ossI, ossO;
    ossI << iInputDim;
    voParsed.insert(voParsed.end(), bpo::option(std::string("input-dim" ), std::vector<std::string>(1, ossI.str())));
    ossO << iOutputDim;
    voParsed.insert(voParsed.end(), bpo::option(std::string("output-dim"), std::vector<std::string>(1, ossO.str())));
  }
  else {
    // no abbreviated dimensions
    ssTokens.seekg(0);
    ssTokens.clear();
  }

  // read other parameters
  std::string sRead;
  short iReadStep = 0;
  std::vector<bpo::option>::iterator iOption;
  do {
    sRead.clear();
    if (iReadStep < 3)
      // read next token (name, equal character or start of value)
      ssTokens >> sRead;
    else {
      // read end of value in quotes
      std::stringbuf sbRead;
      ssTokens.get(sbRead, '\"');
      if (ssTokens.peek() != std::char_traits<char>::eof())
        sbRead.sputc(ssTokens.get());
      sRead = sbRead.str();
    }
    if (sRead.empty() || ssTokens.bad())
      break;
    size_t stPos = 0;
    size_t stLen = sRead.length();

    // read option name
    if (iReadStep <= 0) {
      // new option
      iOption = voParsed.insert(voParsed.end(), bpo::option());

      // option name
      size_t stNPos = sRead.find('=');
      iOption->string_key = sRead.substr(stPos, stNPos - stPos);
      iOption->value.push_back(std::string());

      // next step: read equal character
      iReadStep = 1;
      if (stNPos != std::string::npos)
        stPos = stNPos;
      else
        continue;
    }

    // read equal character
    if (iReadStep == 1) {
      if (sRead[stPos] == '=')
        stPos++;
      else
        // error
        break;

      // next step: read option value
      iReadStep = 2;
      if (stPos >= stLen)
        continue;
    }

    // read option value
    if (iReadStep == 2) {
      // next loop: read new option (if value is not in quotes)
      iReadStep = 0;

      // verify quotes
      size_t stNPos = stLen;
      if (sRead[stPos] == '\"') {
        // option value in quotes
        stPos++;
        if ((stPos < stLen) && (sRead[stLen - 1] == '\"'))
          stNPos--;
        else
          // next loop: end reading value in quotes
          iReadStep = 3;
      }
      iOption->value.back() = sRead.substr(stPos, stNPos - stPos);
      continue;
    }

    // end reading value in quotes
    if (iReadStep >= 3) {
      iOption->value.back().append(sRead, stPos, stLen - stPos - 1);

      // next step: read new option
      iReadStep = 0;
    }
  } while (!(sRead.empty() || ssTokens.bad()));

  // handle errors
  if (ssTokens.bad())
    throw bpo::error("internal stream error");

  return voParsed;
}

/**
 * creates a machine by reading his data from file
 * @param sFileName machine file name
 * @param iBlockSize block size for faster training
 * @param vmMachParams map of parameters read
 * @return new machine object or NULL in case of error
 * @note error code is set if an error occurred
 */
Mach *MachConfig::read_machine_from_file(const std::string &sFileName, int iBlockSize, const bpo::variables_map &vmMachParams)
{
  std::ifstream ifs;
  this->ossErrorInfo.str(sFileName);

  // open file
  ifs.open(sFileName.c_str(), std::ios_base::in);
  if (ifs.fail()) {
    // error handling
    this->eErrorCode = MachConfig::ProbOpenMachineFile;
    return NULL;
  }

  // read file
  Mach *pNewMach = Mach::Read(ifs, iBlockSize);
  if (pNewMach == NULL) {
    // error handling
    this->eErrorCode = MachConfig::ProbAllocMachine;
    return NULL;
  }
  pNewMach->ResetNbEx();

  // apply machine drop-out parameter if defined
  const boost::program_options::variable_value &vvDO = vmMachParams["drop-out"];
  if (!vvDO.empty())
    pNewMach->SetDropOut(vvDO.as<REAL>());

  // initialize MachLin
  MachLin *pMachLin = dynamic_cast<MachLin *>(pNewMach);
  if (pMachLin != NULL) {
    this->apply_machine_parameters(pMachLin, vmMachParams);
    return pNewMach;
  }

  // initialize MachTab
  MachTab *pMachTab = dynamic_cast<MachTab *>(pNewMach);
  if (pMachTab != NULL) {
    this->apply_machine_parameters(pMachTab, vmMachParams);
    return pNewMach;
  }

  return pNewMach;
}

/**
 * applies parameters to given linear machine
 * @note block size parameter is not applied here
 * @param pMachLin pointer to linear machine object
 * @param vmMachParams map of parameters
 * @param bApplyGenVal true to apply general values to parameters as needed, default false otherwise
 */
void MachConfig::apply_machine_parameters(MachLin *pMachLin, const bpo::variables_map &vmMachParams, bool bApplyGenVal) const
{
  if (pMachLin != NULL) {
    bool bWeigthsNotInit = bApplyGenVal;
    bool bBiasNotInit    = bApplyGenVal;

    // constant value for initialization of the weights
    const boost::program_options::variable_value &vvCIW = vmMachParams["const-init-weights"];
    if (!vvCIW.empty()) {
      pMachLin->WeightsConst(vvCIW.as<REAL>());
      bWeigthsNotInit = false;
    }

    // initialization of the weights by identity transformation
    const boost::program_options::variable_value &vvIIW = vmMachParams["ident-init-weights"];
    if (!vvIIW.empty()) {
      pMachLin->WeightsID(vvIIW.as<REAL>());
      bWeigthsNotInit = false;
    }

    // random initialization of the weights by function of fan-in
    const boost::program_options::variable_value &vvFIIW = vmMachParams["fani-init-weights"];
    if (!vvFIIW.empty()) {
      pMachLin->WeightsRandomFanI(vvFIIW.as<REAL>());
      bWeigthsNotInit = false;
    }

    // random initialization of the weights by function of fan-in and fan-out
    const boost::program_options::variable_value &vvFIOIW = vmMachParams["fanio-init-weights"];
    if (!vvFIOIW.empty()) {
      pMachLin->WeightsRandomFanIO(vvFIOIW.as<REAL>());
      bWeigthsNotInit = false;
    }

    // value for random initialization of the weights
    const boost::program_options::variable_value &vvRIW = vmMachParams["random-init-weights"];
    bool bCurRandInitWeights = !vvRIW.empty();
    if (bCurRandInitWeights || bWeigthsNotInit) { // if no init-weights option is used, a general value is applied
      pMachLin->WeightsRandom(bCurRandInitWeights ? vvRIW.as<REAL>() : this->rInitWeights);
    }

    // constant value for initialization of the bias
    const boost::program_options::variable_value &vvCIB = vmMachParams["const-init-bias"];
    if (!vvCIB.empty()) {
      pMachLin->BiasConst(vvCIB.as<REAL>());
      bBiasNotInit = false;
    }

    // value for random initialization of the bias
    const boost::program_options::variable_value &vvRIB = vmMachParams["random-init-bias"];
    bool bCurRandInitBias = !vvRIB.empty();
    if (bCurRandInitBias || bBiasNotInit) { // if no init-bias option is used, a general value is applied
      pMachLin->BiasRandom(bCurRandInitBias ? vvRIB.as<REAL>() : this->rInitBias);
    }
  }
}

/**
 * applies parameters to given table lookup machine
 * @note block size parameter is not applied here
 * @param pMachTab pointer to table lookup machine object
 * @param vmMachParams map of parameters
 * @param bApplyGenVal true to apply general values to parameters as needed, default false otherwise
 */
void MachConfig::apply_machine_parameters(MachTab *pMachTab, const bpo::variables_map &vmMachParams, bool bApplyGenVal) const
{
  if (pMachTab != NULL) {
    bool bTableNotInit = bApplyGenVal;

    // constant value for initialization of the projection layer
    const boost::program_options::variable_value &vvCIP = vmMachParams["const-init-project"];
    if (!vvCIP.empty()) {
      pMachTab->TableConst(vvCIP.as<REAL>());
      bTableNotInit = false;
    }

    // value for random initialization of the projection layer
    const boost::program_options::variable_value &vvRIP = vmMachParams["random-init-project"];
    bool bCurRandInitProj = !vvRIP.empty();
    if (bCurRandInitProj || bTableNotInit) { // if no init-project option is used, a general value is applied
      pMachTab->TableRandom(bCurRandInitProj ? vvRIP.as<REAL>() : this->rInitProjLayer);
    }
  }
}
