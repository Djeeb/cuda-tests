#include "mshadow/tensor.h"
#include <iostream>
#include <random>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

int main(void){
	
	//génération d'un tableau de doubles à allouer
	mt19937 G;
	uniform_real_distribution<double> U(0.,1.)
	n = 1000;
	double data[n];
	for(int i=0;i<n;i++) data[i] = U(G);
	
	//initialisation (obligatoire pour utiliser CuBLAS)
	InitTensorEngine<gpu>();
	
	Tensor<gpu,2> T;
	T.dptr_ = data
	//T = T.slice();
	
	//Arrêt (obligatoire pour utiliser CuBLAS)	
	ShutdownTensorEngine<gpu>();
}
