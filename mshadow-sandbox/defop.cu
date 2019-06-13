#include <cmath>
#include "mshadow/tensor.h"
#include <iostream>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

//_______________________________________________DEFINITION D'OPÉRATEURS

struct tanh{
	MSHADOW_XINLINE static float Map(float x){
		return (exp(2*x)-1)/(exp(2*x)+1);
	}
};

struct ReLu{
	MSHADOW_XINLINE static double Map(double x){
		return (x>0.)?x:0.;
	}
};




int main(void){
	int n = 10;
	InitTensorEngine<gpu>();
	
	//Initialisation du vecteur
	Stream<gpu> * stream_ = NewStream<gpu>(0);
	Tensor<gpu,1, double> Vec = NewTensor<gpu>(Shape1(n), 1., stream_);
	
	//Mapping de la fonction
	cout << "Vec avant mapping :" << endl;
	for(index_t i = 0; i < Vec.size(0); i++) cout << Vec[i] << "\t";	
	Vec = F<tanh>(Vec);
	cout << "\n\nVec après mapping (x -> tanh(x)) :" << endl;
	for(index_t i = 0; i < Vec.size(0); i++) cout << Vec[i] << "\t";
	
	//Nettoyage du device
	FreeSpace(&Vec);
	DeleteStream(stream_);
	
	ShutdownTensorEngine<gpu>();	
}
