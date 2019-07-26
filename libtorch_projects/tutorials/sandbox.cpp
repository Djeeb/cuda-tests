#include <torch/torch.h>
#include <iostream>
using namespace std;


int main(){
	torch::Tensor X = torch::randn({5,3}, torch::dtype(torch::kFloat64));
	torch::Tensor b = torch::ones({2,1}, torch::dtype(torch::kFloat64));
	
	cout << X << endl;
	cout << X.argmax(0) << endl;
}
