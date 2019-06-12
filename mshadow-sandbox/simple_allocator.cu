#include "mshadow/tensor.h"
#include <iostream>
#include <string>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

int main(void){
	
	xpu = "gpu";
	
	//initialisation (obligatoire pour utiliser CuBLAS)
	InitTensorEngine<xpu>();
	
	
	
	
	
	
	
	
	

	//Arrêt (obligatoire pour utiliser CuBLAS)	
	ShutdownTensorEngine<xpu>();
}
