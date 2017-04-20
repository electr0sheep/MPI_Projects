#include <iostream>
#include <random>

using namespace std;

#define N 5
#define BLOCK_SIZE 5
#define RADIUS 1

__global__ void stencil_1d(int *in, int *out){
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;

  // Read input elements into shared memory
  temp[lindex] = in[gindex];
  if (threadIdx.x < RADIUS){
    temp[lindex - RADIUS] = in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }

  // Synchronize (ensure all the data is available)
  __syncthreads();

  // Apply the stencil
  int result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; offset++){
    result += temp[lindex + offset];
  }

  // Store the result
  out[gindex] = result;
}

void random_ints(int *ar, int size){
  int i;

  for (i=0; i<size; i++){
    ar[i] = rand();
  }
}

int main(){
  int *init, *result;
  int *d_init, *d_result;
  int size = N * sizeof(int);
  int i;

  srand(time(NULL));

  // Alloc space for device copies of a, b, c
  cudaMalloc((void **)&d_init, size);
  cudaMalloc((void **)&d_result, size);

  // Alloc space for host copies of a, b, c and setup input values
  init = (int *)malloc(size); random_ints(init, N);
  result = (int *)malloc(size);

  // Copy inputs to device
  cudaMemcpy(d_init, init, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_result, result, size, cudaMemcpyHostToDevice);

  stencil_1d<<<BLOCK_SIZE,N>>>(d_init, d_result);

  // Copy result back to host
  cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

  // Display results
  cout << "INIT" << endl;
  for (i=0; i<N; i++){
    cout << init[i] << " ";
  }
  cout << endl << "RESULT" << endl;
  for (i=0; i<N; i++){
    cout << result[i] << " ";
  }
  cout << endl;

  // Cleanup
  free(init); free(result);
  cudaFree(d_init); cudaFree(d_result);

  return 0;
}
