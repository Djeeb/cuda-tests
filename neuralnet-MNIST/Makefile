CC = nvcc -ccbin g++
CXX = g++
CFLAGS1 = -std=c++11 -O2 -I ../mshadow -lm -lcblas -DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 -DMSHADOW_RABIT_PS=0 -DMSHADOW_DIST_PS=0 -DMSHADOW_USE_PASCAL=0
CFLAGS2 = -std=c++11 -O2 -I ../mshadow -lm -lcblas

nnet_v1:
	$(CC) -c nnet_v1.cu $(CFLAGS1)
	$(CC) -o nnet_v1 nnet_v1.o $(CFLAGS1)
