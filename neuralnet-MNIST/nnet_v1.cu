#include "mshadow/tensor.h"
#include "util.h"
#include <vector>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

int main(){
InitTensorEngine<gpu>();
cout << "------------------------------------------ NEURAL NET V1 ------------------------------------------"

	
	//Chargement des données
	cout << "Chargement des données..." << endl;
	vector<int> Y_train, Y_test;
	TensorContainer<cpu,2> X_train, X_test;
	LoadMNIST("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", Y_train, X_train, true);
	LoadMNIST("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", Y_test, X_test, false);
	
	cout << "Taille des données :" << endl;
	cout << "Train set :" << X_train.shape_ << endl;
	cout << "Test set :" << X_test.shape_ << endl;
	
	
	for(index_t i=0; i<X_train.size(0);i++) cout << X_train[0][i] << endl;



ShutdownTensorEngine<gpu>();
}
