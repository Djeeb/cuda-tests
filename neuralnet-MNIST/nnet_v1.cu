#include "mshadow/tensor.h"
#include "util.h"
#include <vector>

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

int main(){
	
	//Chargement des données
	cout << "Chargement des données..." << endl;
	vector<int> ytrain, ytest;
	TensorContainer<cpu,2> xtrain, xtest;
	LoadMNIST("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", ytrain, xtrain, true);
	LoadMNIST("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", ytest, xtest, false);
	
	cout << xtrain[0][0] << endl;
}
