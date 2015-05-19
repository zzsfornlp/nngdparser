/*
 * CslmInterface.cpp
 *
 *  Created on: 2015Äê3ÔÂ31ÈÕ
 *      Author: zzs
 */

#include "CslmInterface.h"
#include <cmath>
#include <ctime>
#include "../cslm/MachSeq.h"

//tanh table for speedup
REAL* CslmInterface::tanh_table = 0;
int CslmInterface::tanh_table_slots = 100000;	//step 0.0001
void CslmInterface::set_tanh_table()
{
	//[-5,+5], with slots
	REAL start = -5;
	tanh_table = new REAL[tanh_table_slots];
	for(int i=0;i<tanh_table_slots;i++,start += 10.0/tanh_table_slots)
		tanh_table[i] = tanh(start);
}
REAL CslmInterface::tanh_table_tanh(REAL nn)
{
	nn = (nn+5)*tanh_table_slots/10;
	int n = nn;
	if(n<0)
		return -1;
	else if(n>=tanh_table_slots)
		return 1;
	else
		return tanh_table[n];
}

const char* cslm_activations[] = {"Tanh","Nope"};
#define WRITE_CONF_ONE(a1,a2) \
	fout << cslm_activations[parameters->CONF_NN_act] << " = " << (int)(a1) << "x" << (int)(a2) << " fanio-init-weights=1.0\n";
static inline void write_conf_no_split(parsing_conf* parameters,int dict_count,int xdim,int outdim)
{
	//--first write conf
	ofstream fout(parameters->CONF_mach_conf_name.c_str());
	fout << "block-size = " << parameters->CONF_NN_BS << "\n";
	if(parameters->CONF_NN_drop>0)
		fout << "drop-out = " << parameters->CONF_NN_drop << "\n";
	int width = xdim*parameters->CONF_NN_we;
	//1.projection layer
	fout << "[machine]\n" << "Sequential = \n" << "Parallel = \n";
	for(int i=0;i<xdim;i++)
		fout << "Tab = " << dict_count << "x" << parameters->CONF_NN_we << "\n";
	fout << "#End\n";
	//2.hidden layers
	for(int i=0;i<parameters->CONF_NN_plus_layers;i++){
		WRITE_CONF_ONE(width,parameters->CONF_NN_h_size[i]);
		width = parameters->CONF_NN_h_size[i];
	}
	//3.output layer
	if(outdim>1){
		//output multiclass-class(0 or 1)
		fout << "Softmax = " << width << "x" << outdim << " fanio-init-weights=1.0\n";
	}
	else{
		//linear output score
		fout << "Linear = " << width << "x" << 1 << " fanio-init-weights=1.0\n";
	}
	fout << "#End\n";
	fout.close();
}
static inline void write_conf_split(parsing_conf* parameters,int dict_count,int xdim,int outdim,int split_num)
{
	//here only support CONF_NN_h_size and must have symmetric input
	ofstream fout(parameters->CONF_mach_conf_name.c_str());
	fout << "block-size = " << parameters->CONF_NN_BS << "\n";
	if(parameters->CONF_NN_drop>0)
		fout << "drop-out = " << parameters->CONF_NN_drop << "\n";
	int width = xdim*parameters->CONF_NN_we;
	//1.projection layer
	fout << "[machine]\n" << "Sequential = \n" << "Parallel = \n";
	for(int i=0;i<xdim;i++)
		fout << "Tab = " << dict_count << "x" << parameters->CONF_NN_we << "\n";
	fout << "#End\n";
	//2.then
	//hidden layer1 -- split
	{
		int each_1 = width / split_num;
		int each_2 = parameters->CONF_NN_h_size[0] / split_num;		//should be divided
		fout << "Parallel = \n";
		for(int i=0;i<split_num;i++)
			WRITE_CONF_ONE(each_1,each_2);
		fout << "#End\n";
	}
	//more hidden layers??
	for(int i=0;i<parameters->CONF_NN_plus_layers-1;i++){
		WRITE_CONF_ONE(parameters->CONF_NN_h_size[i],parameters->CONF_NN_h_size[i+1]);
	}
	width = parameters->CONF_NN_h_size[parameters->CONF_NN_plus_layers-1];
	//3.output
	if(outdim>1)
		fout << "Softmax = " << width << "x" << outdim << " fanio-init-weights=1.0\n";
	else
		fout << "Linear = " << width << "x" << 1 << " fanio-init-weights=1.0\n";
	fout << "#End\n";
	fout.close();
}
void CslmInterface::mach_split_share()
{
	//specified sharing first layer(after projection)
	//	--- don't care memory problems because we never delete(fortunately memory is enough and deleting is at the last...)
	MachMulti* m = (MachMulti*)mach;	//seq
	m = (MachMulti*)(m->MachGet(1));	//par2
	MachLin* mm = (MachLin*)(m->MachGet(0));	//lin
	REAL* ww = mm->w;
	REAL* bb = mm->b;
	for(int i=1;i<m->MachGetNb();i++){
		((MachLin*)(m->MachGet(i)))->w = ww;
		((MachLin*)(m->MachGet(i)))->b = bb;
	}
}

//specified init
CslmInterface* CslmInterface::create_one(parsing_conf* parameters,FeatureGen* f,int outdim)
{
	//1.conf file
	if(parameters->CONF_NN_split)
		write_conf_split(parameters,f->get_dict()->get_count(),f->get_xdim(),outdim,f->get_order()+1);
	else
		write_conf_no_split(parameters,f->get_dict()->get_count(),f->get_xdim(),outdim);
	//2. get machine
	MachConfig mach_config(true);
	//for mach_config
	char *argv[2];
	argv[0] = "nn";
	argv[1] = (char*)parameters->CONF_mach_conf_name.c_str();
	mach_config.parse_options(2,argv);
    Mach* mach = mach_config.get_machine();
    if(mach == 0)
    	Error(mach_config.get_error_string().c_str());
    CslmInterface* ret = new CslmInterface(mach);
    //3.split??
    if(parameters->CONF_NN_split && parameters->CONF_NN_split_share)
    	ret->mach_split_share();
    return ret;
}

CslmInterface* CslmInterface::Read(string name){
	ifstream ifs;
	ifs.open(name.c_str(),ios::binary);
	Mach* m = Mach::Read(ifs);
	ifs.close();
	string file2 = name + CSLM_MACHINE_DESCRIBE_SUFFIX;
	ifs.open(file2.c_str(),ios::binary);
	REAL* pre_calc_table=0;
	if(ifs){	//if pre_calc file exists
		long size = 0;
		ifs.read((char*)&size,sizeof(long));
		pre_calc_table = new REAL[size];
		ifs.read((char*)pre_calc_table,size*sizeof(REAL));
		ifs.close();
	}
	if(m)
		return new CslmInterface(m,pre_calc_table);
	else
		return 0;
}

void CslmInterface::Write(string name){
	ofstream fs;
	fs.open(name.c_str(),ios::binary);
	mach->Write(fs);
	fs.close();
	if(pre_calc_table){ //not zero
		string file2 = name + CSLM_MACHINE_DESCRIBE_SUFFIX;
		fs.open(file2.c_str(),ios::binary);
		//continue reading
		fs.write((char*)&pre_calc_size,sizeof(long));
		fs.write((char*)pre_calc_table,pre_calc_size*sizeof(REAL));
		fs.close();
	}
}

REAL* CslmInterface::mach_forward(REAL* assign,int all)
{
	//specified mach structures
	MachSeq* m1 = (MachSeq*)mach;
	Mach* m2 = m1->MachGet(2);	//0 is tab,1 is linear
	MachLin* m11 = (MachLin*)m1->MachGet(1);	//0 is tab,1 is linear+tanh
	REAL* b = m11->b;		//nope, don't need it here
	//forward
	Mach* m = mach;
	int idim = m->GetIdim();
	int odim = m->GetOdim();
	int remain = all;
	int bsize = m->GetBsize();
	REAL* xx = assign;
	REAL* mach_y = new REAL[all*odim];
	REAL* yy = mach_y;
	while(remain > 0){
		int n=0;
		if(remain >= bsize)
			n = bsize;
		else
			n = remain;
		remain -= bsize;
		m->SetDataIn(xx);
		if(pre_calc_table){
			//short cut
			REAL* data_in = m2->GetDataIn();
			REAL* assign_data_in = data_in;
			for(int i=0;i<n;i++){
				memcpy(assign_data_in+i*second_layer_dim,b,sizeof(REAL)*second_layer_dim);
			}
			REAL* words_indexes = xx;
			for(int i=0;i<n;i++){
				for(int j=0;j<embed_layer_num;j++){
					REAL* to_assign_base = pre_calc_table+second_layer_dim*embed_layer_num*((int)words_indexes[j]);
					REAL* to_assign = to_assign_base + j*second_layer_dim;
					for(int d=0;d<second_layer_dim;d++)
						assign_data_in[d] += to_assign[d];
				}
				words_indexes += embed_layer_num;
				assign_data_in += second_layer_dim;
			}
			//tanh --- currently only support tanh
			for(int i=0;i<n*second_layer_dim;i++){
				//data_in[i] = tanh(data_in[i]);
				data_in[i] = tanh_table_tanh(data_in[i]);
			}
			//forward
			for(int i=2;i<m1->MachGetNb();i++){
				Mach* tmp = m1->MachGet(i);
				tmp->Forw(n);
			}
		}
		else
			m->Forw(n);
		memcpy(yy, m->GetDataOut(), odim*sizeof(REAL)*n);
		yy += n*odim;
		xx += n*idim;
	}
	return mach_y;
}

//example: 200*44*30000 second_layer_dim*embed_layer_num*dict_num
void CslmInterface::pre_calc()
{
	//specified mach structures
	MachSeq* m1 = (MachSeq*)mach;
	REAL* tabs = get_tab();
	MachLin* m11 = (MachLin*)m1->MachGet(1);	//0 is tab,1 is linear+tanh
	REAL* w = m11->w;
	//REAL* b = m11->b;		//nope, don't need it here

	long size = second_layer_dim*embed_layer_num*dict_num;
	pre_calc_table = new REAL[size];

	time_t now;
	time(&now);
	cout << "##Start pre-calculating with REAL of " << size << " at " << ctime(&now) << flush;
	REAL* assign_output = pre_calc_table;
	int m11_idim = m11->GetIdim();	//m11_idim must be embed_layer_num*embed_dim
	if(m11_idim != embed_layer_num*embed_dim)
		nnError(NNERROR_InternalError);
	REAL* once_input = new REAL[m11_idim*embed_layer_num];	//some kind of sparse
	for(int i=0;i<m11->GetIdim()*embed_layer_num;i++)
		once_input[i] = 0;
	for(int i=0;i<dict_num;i++){
		REAL* tab_assign = tabs+i*embed_dim;
		for(int j=0;j<embed_layer_num;j++)
			memcpy(once_input+j*m11_idim+j*embed_dim,tab_assign,embed_dim*sizeof(REAL));
		call_gemm(assign_output,w,once_input,0.0,second_layer_dim,embed_layer_num,m11_idim);
		assign_output += second_layer_dim*embed_layer_num;
	}
	time(&now);
	cout << "##Finish pre-calculating with REAL of " << size << " at " << ctime(&now) << flush;
}

void CslmInterface::DEBUG_pre_calc()
{
	MachSeq* m1 = (MachSeq*)mach;
	MachLin* m11 = (MachLin*)m1->MachGet(1);	//0 is tab,1 is linear+tanh
	MachLin* m12 = (MachLin*)m1->MachGet(2);	//0 is tab,1 is linear+tanh
	REAL* b = m11->b;
	int m11_idim = m11->GetIdim();

	int test_num = 10;
	REAL* precalc_output = new REAL[this->GetWidth()*second_layer_dim];
	REAL* real_input = new REAL[this->GetWidth()*embed_layer_num];
	for(int t=0;t<embed_layer_num*embed_layer_num;t++)
		real_input[t] = -1;

	//1.phase1 --- debug table
	/*  //---check, only small differences maybe caused by numerical reasons
	for(int i=0;i<test_num;i++){
		int pick_word = (rand())%dict_num;
		cout << "Test1, pick word num " << pick_word << endl;
		//1.1 precalc
		memcpy(precalc_output,pre_calc_table+pick_word*embed_layer_num*second_layer_dim,sizeof(REAL)*embed_layer_num*second_layer_dim);
		for(int t1=0;t1<embed_layer_num;t1++)
			for(int t2=0;t2<second_layer_dim;t2++)
				precalc_output[t1*second_layer_dim+t2] += b[t2];
		for(int t=0;t<embed_layer_num*second_layer_dim;t++)
			precalc_output[t] = tanh(precalc_output[t]);
		//1.2 real forward
		for(int t=0;t<embed_layer_num;t++)
			real_input[t*embed_layer_num+t] = pick_word;
		mach->SetDataIn(real_input);
		mach->Forw(embed_layer_num);
		//1.3 diff
		REAL* real_ouput = m12->GetDataIn();
		int wrong = 0;
		int wrong_large = 0;
		REAL all_wrong = 0;
		for(int t=0;t<embed_layer_num*second_layer_dim;t++){
			REAL diff_2 = precalc_output[t] - real_ouput[t];
			if(diff_2 < 0)
				diff_2 = - diff_2;
			if(precalc_output[t] != real_ouput[t])
				wrong++;
			if(diff_2 > 0.0001)
				wrong_large++;
			all_wrong += diff_2;
		}
		cout << all_wrong << "," << wrong_large << "," << wrong << "/" << embed_layer_num*second_layer_dim << endl;
	}*/

	//2.phase2 --- forward one instance
	/* //also check
	for(int i=0;i<test_num;i++){
		int* pick_word = new int[embed_layer_num];
		cout << "Test1, pick word nums" << endl;
		for(int t=0;t<embed_layer_num;t++)
			pick_word[t] = (rand())%dict_num;
		//1.1 precalc
		for(int t=0;t<second_layer_dim;t++)
			precalc_output[t] = b[t];
		for(int t=0;t<embed_layer_num;t++){
			REAL* assign = pre_calc_table + pick_word[t]*embed_layer_num*second_layer_dim + t*second_layer_dim;
			for(int t2=0;t2<second_layer_dim;t2++)
				precalc_output[t2] += assign[t2];
		}
		for(int t=0;t<second_layer_dim;t++)
			precalc_output[t] = tanh(precalc_output[t]);
		//1.2 real forward
		for(int t=0;t<embed_layer_num;t++)
			real_input[t] = pick_word[t];
		mach->SetDataIn(real_input);
		mach->Forw(1);
		//1.3 diff
		REAL* real_ouput = m12->GetDataIn();
		int wrong = 0;
		int wrong_large = 0;
		REAL all_wrong = 0;
		for(int t=0;t<embed_layer_num;t++){
			REAL diff_2 = precalc_output[t] - real_ouput[t];
			if(diff_2 < 0)
				diff_2 = - diff_2;
			if(precalc_output[t] != real_ouput[t])
				wrong++;
			if(diff_2 > 0.0001)
				wrong_large++;
			all_wrong += diff_2;
		}
		cout << all_wrong << "," << wrong_large << "," << wrong << "/" << embed_layer_num << endl;
		delete []pick_word;
	}*/

	//3.phase3 --- forward bs
	int bs = this->GetWidth();
	for(int i=0;i<test_num;i++){
		int* pick_word = new int[embed_layer_num*bs];
		cout << "Test1, pick word nums" << endl;
		for(int t=0;t<embed_layer_num*bs;t++)
			pick_word[t] = (rand())%dict_num;
		//1.1 precalc
		for(int t=0;t<second_layer_dim*bs;t++)
			precalc_output[t] = b[t%second_layer_dim];
		for(int t=0;t<bs;t++){
			REAL* to_asssign = precalc_output + t*second_layer_dim;
			for(int t2=0;t2<embed_layer_num;t2++){
				REAL* assign = pre_calc_table + pick_word[t2+t*embed_layer_num]*embed_layer_num*second_layer_dim + t2*second_layer_dim;
				for(int t3=0;t3<second_layer_dim;t3++)
					to_asssign[t3] += assign[t3];
			}
		}
		for(int t=0;t<second_layer_dim*bs;t++)
			precalc_output[t] = tanh(precalc_output[t]);
		//1.2 real forward
		for(int t=0;t<embed_layer_num*bs;t++)
			real_input[t] = pick_word[t];
		mach->SetDataIn(real_input);
		mach->Forw(bs);
		//1.3 diff
		REAL* real_ouput = m12->GetDataIn();
		int wrong = 0;
		int wrong_large = 0;
		REAL all_wrong = 0;
		for(int t=0;t<embed_layer_num*bs;t++){
			REAL diff_2 = precalc_output[t] - real_ouput[t];
			if(diff_2 < 0)
				diff_2 = - diff_2;
			if(precalc_output[t] != real_ouput[t])
				wrong++;
			if(diff_2 > 0.0001)
				wrong_large++;
			all_wrong += diff_2;
		}
		cout << all_wrong << "," << wrong_large << "," << wrong << "/" << embed_layer_num*bs << endl;
		delete []pick_word;
	}
}
