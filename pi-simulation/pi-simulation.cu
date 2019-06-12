#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <chrono>

using namespace std;


 //Fonction d'équation du cercle avec __global__ pour utiliser sur host ou device
__global__ void
cercle(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i]*A[i] + B[i]*B[i];
    }
}

int main(void) {
	
	//Initialisatio
	int numElements;
	cout << "---Estimation de pi par méthode de monte carlo sur GPU---" << endl;
    cout << "Entrez le nombre de simulations que vous voulez faire : " << endl;
    cin >> numElements;
   
 
	//allocation de la taille des vecteurs dans le host (CPU)
	size_t size = numElements * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);


    // simulation
    cout << "simulation..." << endl;
    mt19937 G;
    uniform_real_distribution<float> U(-1.,1.);
    for (int i = 0; i < numElements; ++i)
		{
        h_A[i] = U(G);
        h_B[i] = U(G);
		}


    // allocation de la taille des vecteurs dans le device (GPU)
    float *d_A = NULL;
    cudaMalloc((void **)&d_A, size);
    float *d_B = NULL;
    cudaMalloc((void **)&d_B, size);
    float *d_C = NULL;
    cudaMalloc((void **)&d_C, size);


    // Copie host -> device (input)
    cout << "Copie host -> device (input)" << endl;
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    // Lancement de la tâche
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("Le noyau CUDA démarre la tâche avec %d blocs de %d threads\n", blocksPerGrid, threadsPerBlock);
    auto t1 = chrono::system_clock::now();
	cercle<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
	auto t2 = chrono::system_clock::now();
	
    // Copie device -> host (output)
    cout << "Copie device -> host (output)" << endl;
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    
    // Vérification
    float est_pi =0.;
    for (int i = 0; i < numElements; ++i) est_pi += (h_C[i]<1)?1.:0.;
	cout << "estimation de pi : " << (est_pi*4)/float(numElements) << " pour une simulation de taille " << numElements << endl;


    // Libération de la mémoire device...
    cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	
    // Libération de la mémoire host ...
    free(h_A);
    free(h_B);
    free(h_C);


    // Reset du device
    cudaDeviceReset();
    
    
	
	chrono::duration<double> diff = t2 - t1;
    cout << "Temps de calcul de la tâche : " << diff.count() << endl;
    return 0;
}
