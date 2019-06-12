#include "mshadow/tensor.h"
#include <iostream>
#include <random>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

int main(void){
	
	//génération d'un tableau de doubles à allouer
	mt19937 G;
	uniform_real_distribution<double> U(0.,1.);
	int n = 1000;
	double * data[n];
	for(int i=0;i<n;i++) data[i] = U(G);
	
	//initialisation (obligatoire pour utiliser CuBLAS)
	InitTensorEngine<gpu>();
	
	Tensor<gpu,2, double> T;
	
	//allocation du tableau de double
	cout << "Allocation d'un tableau de " << n << " valeurs sur dans un tenseur :" << endl;
	T.dptr_ = data;
	
	//test sur les dimensions
	cout << "size  :" << T.size(0) << endl;
	cout << "shape : " << T.shape_ << endl;
	cout << "\nTenseur redimensionné au maximum en utilisant la méthode .Slice() :" << endl;
	T = T.Slice(0,n);
	cout << "size  : " << T.size(0) << endl;
	cout << "shape : " << T.shape_ << endl;
	
	//Arrêt (obligatoire pour utiliser CuBLAS)	
	ShutdownTensorEngine<gpu>();
}
