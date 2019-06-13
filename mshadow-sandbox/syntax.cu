#include "mshadow/tensor.h"
#include <iostream>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

int main(void){
	int n = 100;
	
	cout << "\n--- Exploration de la syntaxe de mshadow ---" << endl;
	
	//Initialisation du tensor Engine pour CuBLAS
	InitTensorEngine<cpu>();
	
	//génération de données
	double data[n];
	for(int i=0;i<n;i++) data[i] = i;
	
	//création d'un tenseur 
	Tensor<cpu,2,double> T( data , Shape2(10,10));
	
	Tensor<cpu,1,double> Vec = T[0]; 
	
	cout << "\nT est une matrice " << T.size(0) << "x" << T.size(1) << " : " << endl;
	
	for(index_t i = 0; i < T.size(0); i++){
		for(index_t j = 0; j < T.size(1); j++){
			cout << T[i][j] << "\t";
		} 
		cout << "\n";	
	}
	
	cout << "\nVec est un vecteur de taille " << Vec.size(0) << " copié sur T[0] : " << endl;
	for(index_t i = 0; i < T.size(0); i++) cout << Vec[i] << "\t";
		
	//Modification rapide de la matrice
	T = 1.;
	cout << "\n\nAprès modification de la matrice, T est maintenant égal à : " << endl;
	for(index_t i = 0; i < T.size(0); i++){
		for(index_t j = 0; j < T.size(1); j++){
			cout << T[i][j] << "\t";
		} 
		cout << "\n";	
	}
	
	cout << "\nLe vecteur est quant à lui égal à : " << endl;
	for(index_t i = 0; i < T.size(0); i++) cout << Vec[i] << "\t";	
	
	//Modification du vecteur par lazy evaluation
	for(int i=1; i < 10; i++) Vec += T[i];
	
	cout << "\n\nPuis après addition de toutes les lignes dans Vec : " << endl;
	for(index_t i = 0; i < T.size(0); i++) cout << Vec[i] << "\t";	
	
	
	
	//Fermeture du tensor Engine
	ShutdownTensorEngine<cpu>();
	cout << "\n";
	
}
