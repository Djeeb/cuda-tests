#include "mshadow/tensor.h"
#include <iostream>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

int main(void){
	
	//initialisation (obligatoire pour utiliser CuBLAS)
	InitTensorEngine<cpu>();
	
	
	
	
	
	
	
	
	

	//Arrêt (obligatoire pour utiliser CuBLAS)	
	ShutdownTensorEngine<cpu>();
}
