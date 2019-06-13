#include "mshadow/tensor.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
class Vector;

using namespace mshadow;
using namespace mshadow::expr;
using namespace std;

class Vector{
	public:
		int size;
		double * data;
		
		Vector(int n=0);
		~Vector(){delete [] data;}
		
		double operator[](int i) const {return data[i];}
		double & operator[](int i) {return data[i];}
		
		friend Vector operator+(const Vector &, const Vector &);
		Vector operator=(const Vector &);
			
};

Vector::Vector(int n){
	data = new double[n];
	for(int i=0; i<n; i++) data[i] = 0.; 
}

Vector Vector::operator=(const Vector & A){
	if(this == &A){return *this;}
	else{
		delete [] data;
		data = new double[A.size];
		for(int i=0; i<size; i++) data[i] = A.data[i]; 
	}
	return *this;
}

Vector operator+(const Vector & A, const Vector & B){
	Vector C(A.size);
	for(int i=0; i < C.size; i++) C[i] = A[i] + B[i];
	return C;
}


int main(void){
	
	
	InitTensorEngine<gpu>();

	//initialisation
	mt19937 G;
	uniform_real_distribution<double> U(-1.,1.);
	int n = 1000000;
	Vector A(n), B(n), C(n),S(n);
	for(int i = 0; i < n; i++){
		A[i] = U(G);
		B[i] = U(G);
		C[i] = U(G);
	}
	cout << "--- Calcul de A + B + C (3 vecteurs de taille " << n << ") par différentes méthodes ---" << endl;
	
	
	//méthode naïve
	auto t1 = chrono::system_clock::now();
	S = A + B + C;
	auto t2 = chrono::system_clock::now();
	chrono::duration<double> diff = t2 - t1;
	cout << " \nTemps de calcul méthode naïve :" << diff.count() << endl;
	
	ShutdownTensorEngine<gpu>();

}
