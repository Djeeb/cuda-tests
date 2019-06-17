#include <torch/torch.h>
#include <iostream>

using namespace std;

int main(){
	at::Tensor T = torch::rand({3,3});
	at::Tensor Id = torch::ones({3,3});
	
	cout << T << endl;
	cout << Id << endl;
	
}
