CC = nvcc
CFLAGS = -ccbin g++ -std=c++11 -O2

scalaire-test: scalaire-test.o
	$(CC) -o scalaire scalaire-test.o $(CFLAGS)
	
scalaire-test.o: scalaire-test.cu
	$(CC) -c scalaire-test.cu $(CFLAGS)

clean:
	rm scalaire-test.o
	rm scalaire-test
