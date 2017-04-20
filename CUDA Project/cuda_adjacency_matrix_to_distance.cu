#include <iostream>

using namespace std;

#define N 10
#define BLOCK_SIZE 8
#define THREAD_COUNT 32

void initializeCircularGraph(int*);
void initializeLinearGraph(int*);
void writeMatrix(int*, int, int, int, int);
void printMatrix(int*);

__global__ void calculateDistance(int *matrix){

}

int main(){
  int *matrix;
  int *d_matrix;
  int size = N * N * sizeof(int);

  // Allocate space for device copy of matrix
  cudaMalloc((void **)&d_matrix, size);

  // Allocate space for host copy of matrix and initialize
  matrix = (int *)malloc(size); initializeCircularGraph(matrix);

  // Copy matrix to device
  cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice);

  // do cuda stuff
  calculateDistance<<<BLOCK_SIZE,THREAD_COUNT>>>(d_matrix);

  cout << "                          ADJACENCY MATRIX" << endl;
  cout << "==============================================================================" << endl;
  printMatrix(matrix);

  // Copy results back to host
  cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);

  cout << "                            DISTANCE MATRIX" << endl;
  cout << "==============================================================================" << endl;
  printMatrix(matrix);

  // Cleanup
  free(matrix);
  cudaFree(d_matrix);

  return 0;
}

void initializeCircularGraph(int *ar){
  int i, j;

  for (j=0; j<N; j++){
    for (i=0; i<N; i++){
      if (j == i+1 || i == j+1){
        writeMatrix(ar, i, j, N, 1);
      } else if ((j == N-1 && i == 0) || (j == 0 && i == N-1)){
        writeMatrix(ar, i, j, N, 1);
      } else{
        writeMatrix(ar, i, j, N, 999999);
      }
    }
  }

  for (i=0; i<N; i++){
    writeMatrix(ar, i, i, N, 0);
  }
}

void initializeLinearGraph(int *ar){
  int i, j;

  for (j=0; j<N; j++){
    for (i=0; i<N; i++){
      if (j == i+1 || i == j+1){
        writeMatrix(ar, i, j, N, 1);
      } else {
        writeMatrix(ar, i, j, N, 999999);
      }
    }
  }

  for (i=0; i<N; i++){
    writeMatrix(ar, i, i, N, 0);
  }
}

void writeMatrix(int *matrix, int x, int y, int width, int value){
  matrix[x+(y*width)] = value;
}

void printMatrix(int *matrix){
  int i;

  for (i=0; i<N*N; i++){
    if (i%N == 0){
      cout << endl;
    }
    cout << matrix[i] << "\t";
  }
  cout << endl;
}
