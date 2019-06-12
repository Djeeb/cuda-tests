#include "mshadow/tensor.h"
#include <iostream>
#include <string>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

int main(void){
	
	//choix GPU/CPU
	int choix;
	cout << "Quel mode désirez vous ? Entrez le chiffre correspondant :" << endl;
	cout << "1.CPU \t 2.GPU" << endl;
	cin >> choix;
	auto xpu = (choix==1)?cpu:gpu;
	
	//initialisation (obligatoire pour utiliser CuBLAS)
	InitTensorEngine<xpu>();
	
	
	
	
	
	
	
	
	

	//Arrêt (obligatoire pour utiliser CuBLAS)	
	ShutdownTensorEngine<xpu>();
}
