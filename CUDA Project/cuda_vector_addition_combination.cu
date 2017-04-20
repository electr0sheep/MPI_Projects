#include <iostream>
#include <random>

using namespace std;

#define N (2048*2048)
#define THREADS_PER_BLOCK 512

__global__ void add(int *a, int *b, int *c){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}

void random_ints(int *ar, int size){
  int i;

  for (i=0; i<size; i++){
    ar[i] = rand();
  }
}

int main(){
  int *a, *b, *c;            // host copies of a, b, c
  int *d_a, *d_b, *d_c;      // devices copies of a, b, c
  int size = N * sizeof(int);

  srand(time(NULL));

  // Alloc space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Alloc space for host copies of a, b, c and setup input values
  a = (int *)malloc(size); random_ints(a, N);
  b = (int *)malloc(size); random_ints(b, N);
  c = (int *)malloc(size);

  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU with N blocks
//****************************************************************************//
//    THIS IS THE DIFFERENCE BETWEEN THE BLOCKS VERSION AND THREAD VERSION    //
//****************************************************************************//
  add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  // Displaying results in this case will take too long

  // Cleanup
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  cout << "End Program" << endl;

  return 0;
}
