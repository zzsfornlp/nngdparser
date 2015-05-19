/*
 * EisnerO3g.h
 *
 *  Created on: 2015.4.20
 *      Author: zzs
 */

#ifndef ALGORITHMS_EISNERO3G_H_
#define ALGORITHMS_EISNERO3G_H_

#include <iostream>
#include <cstdlib>
#include <vector>
using namespace std;

// ----------- for decoding o3g ----------------

#define Negative_Infinity_O3g -1e100
#define E_INCOM_O3g 0
#define E_COM_O3g 1
#define E_SIB_O3g 2

//the index explanation --- C[len][len][len][3]
// -- slightly different than the previous ones
inline int get_index_o3g(int len,int g,int h,int m,int c)
{
	int key = g;
	key = key * len + h;
	key = key * len + m;
	key = key * 3 + c;
	return key;
}

inline int get_index2_o3g(int len,int g,int h,int c,int m)
{
	//get for scores: S[g][h][c][m] (S[0][0][c][m];S[g][h][-][m]=>S[g][h][h][m])
	int key = g;
	key = key * len + h;
	key = key * len + c;
	key = key * len + m;
	return key;
}

extern vector<int>* decodeProjective_o3g(int length,double* scores);




#endif /* ALGORITHMS_EISNERO3G_H_ */
