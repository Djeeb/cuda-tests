CC = nvcc
CFLAGS = -O3 --use_fast_math -ccbin g++  -Xcompiler "-std=c++11 -Wall -O3 -I../mshadow -fopenmp -funroll-loops -Wno-unused-parameter -Wno-unknown-pragmas -Wno-unused-local-typedefs -msse3 -mf16c -I/opt/cuda/include -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=1 -DMSHADOW_RABIT_PS=0 -DMSHADOW_DIST_PS=0 -DMSHADOW_USE_PASCAL=0" -Xlinker "-lm -lcblas -lblas -lcudart -lcublas -lcurand -lcusolver -L/opt/cuda/lib64 -L/opt/cuda/lib -lopenblas"


nnet_v1: nnet_v1.cu neuralnet.hpp
	$(CC) -o nnet_v1 $(CFLAGS) nnet_v1.cu

clean:
	rm nnet_v1
