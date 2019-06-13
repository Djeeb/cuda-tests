#include <cmath>
#include "mshadow/tensor.h"
#include <iostream>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

//_______________________________________________DEFINITION D'OPÉRATEURS

struct tanh{
	template<typename type>
	MSHADOW_XINLINE static type Map(type x){
		return (exp(2*x)-1)/(exp(2*x)+1);
	}
};

struct ReLu{
	template<typename type>
	MSHADOW_XINLINE static type Map(type x){
		return (x>0.)?x:0.;
	}
};

struct addone {
  // map can be template function
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return  a + static_cast<DType>(1);
  }
};


int main(void){
	int n = 10;
	InitTensorEngine<cpu>();
	
	//Initialisation du vecteur
	Stream<cpu> * stream_ = NewStream<cpu>(0);
	Tensor<cpu,1, double> Vec = NewTensor<cpu>(Shape1(n), 1., stream_);
	
	//Mapping de la fonction
	cout << "Vec avant mapping :" << endl;
	for(index_t i = 0; i < Vec.size(0); i++) cout << Vec[i] << "\t";	
	Vec = F<addone>(Vec);
	cout << "\n\nVec après mapping (x -> tanh(x)) :" << endl;
	for(index_t i = 0; i < Vec.size(0); i++) cout << Vec[i] << "\t";
	
	//Nettoyage du device
	FreeSpace(&Vec);
	
	ShutdownTensorEngine<cpu>();	
}
