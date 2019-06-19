#include <torch/torch.h>
#include <iostream>
#include <chrono>

using namespace std;

int main(){
//Paramètres
	torch::Device host(torch::kCPU);
	torch::Device device(torch::kCUDA);
	int n;
	cout << "Entrez le nombre d'estimations désiré" << endl;
	cin >> n;
	cout << "--- Estimation de pi sur CPU vs GPU pour n = " << n << " ---" << endl;

//Estimation sur CPU	
	auto t1 = chrono::system_clock::now();
	torch::Tensor x = torch::rand({n},host);
	torch::Tensor y = torch::rand({n},host);
	auto Est = - floor(x*x + y*y - 1);
	auto Sum = at::sum(Est)/double(n);
	auto t2 = chrono::system_clock::now();
	
	cout << "\n--- Sur CPU : " << endl;
	cout << "estimation : " << Sum.item<float>()*4 << endl;
	chrono::duration<double> diff = t2 - t1;
	cout << "temps de calcul : " << diff.count() << " sec" <<  endl;



	if (torch::cuda::is_available()) {
	cout << "\n\nCUDA est disponible! Entraînement sur GPU... \n\n" << endl;
}


//Estimation sur GPU
	auto t3 = chrono::system_clock::now();
	torch::Tensor x2 = torch::rand({n},device);
	torch::Tensor y2 = torch::rand({n},device);
	auto Est2 = (-floor(x2*x2 + y2*y2 - 1));
	auto Sum2 = (at::sum(Est2)/double(n));
	auto t4 = chrono::system_clock::now();

	cout << "\n--- Sur GPU : " << endl;
	cout << "estimation : " << Sum2.item<float>()*4 << endl;
	chrono::duration<double> diff2 = t4 - t3;
	cout << "temps de calcul : " << diff2.count() << " sec" <<  endl;
}
