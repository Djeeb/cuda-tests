#include "mshadow/tensor.h"
#include <iostream>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

int main(void){
	int n = 100;
	
	cout << "--- Exploration de la syntaxe de mshadow ---" << endl;
	
	//Initialisation du tensor Engine pour CuBLAS
	InitTensorEngine<gpu>();
	
	//génération de données
	double data[n];
	for(int i=0;i<n;i++) data[i] = i;
	
	//création d'un tenseur 
	Tensor<gpu,2,double> T( data , Shape2(10,10));
	
	Tensor<gpu,1,double> Vec = T[0]; 
	
	cout << "T est une matrice " << T.size(0) << "x" << T.size(1) << " : " << endl;
	
	for(index_t i = 0;i < T.size(0) < i++){
		for(index_t j = 0;j < T.size(1) < j++){
			cout << T[i][j] << "\ŧ";
		} 
		cout << "\n";	
	}
}
