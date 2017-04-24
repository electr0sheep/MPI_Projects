#include <iostream>
#include <fstream>

using namespace std;

#define N 256
#define BLOCK_SIZE 8
#define THREAD_COUNT 32

void initializeCircularGraph(int*);
void initializeLinearGraph(int*);
void printMatrix(int*, ofstream&);
__host__ __device__ void writeMatrix(int*, int, int, int);
__device__ int readMatrix(int*, int, int);
__global__ void calculateDistanceMatrix(int*);

int main(){
  int *matrix;
  int *d_matrix;
  int size = N * N * sizeof(int);
  ofstream myFile;

  myFile.open("output.txt");

  // Allocate space for device copy of matrix
  cudaMalloc((void **)&d_matrix, size);

  // Allocate space for host copy of matrix and initialize
  matrix = (int *)malloc(size); initializeCircularGraph(matrix);

  // Copy matrix to device
  cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice);

  // do cuda stuff
  calculateDistanceMatrix<<<BLOCK_SIZE,THREAD_COUNT>>>(d_matrix);

  myFile << "                          ADJACENCY MATRIX" << endl;
  myFile << "==============================================================================" << endl;
  printMatrix(matrix, myFile);

  // Copy results back to host
  cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);

  myFile << "                            DISTANCE MATRIX" << endl;
  myFile << "==============================================================================" << endl;
  printMatrix(matrix, myFile);

  // Cleanup
  free(matrix);
  cudaFree(d_matrix);
  myFile.close();

  return 0;
}

void initializeCircularGraph(int *ar){
  int i, j;

  for (j=0; j<N; j++){
    for (i=0; i<N; i++){
      if (j == i+1 || i == j+1){
        writeMatrix(ar, i, j, 1);
      } else if ((j == N-1 && i == 0) || (j == 0 && i == N-1)){
        writeMatrix(ar, i, j, 1);
      } else{
        writeMatrix(ar, i, j, 999999);
      }
    }
  }

  for (i=0; i<N; i++){
    writeMatrix(ar, i, i, 0);
  }
}

void initializeLinearGraph(int *ar){
  int i, j;

  for (j=0; j<N; j++){
    for (i=0; i<N; i++){
      if (j == i+1 || i == j+1){
        writeMatrix(ar, i, j, 1);
      } else {
        writeMatrix(ar, i, j, 999999);
      }
    }
  }

  for (i=0; i<N; i++){
    writeMatrix(ar, i, i, 0);
  }
}

void printMatrix(int *matrix, ofstream &myFile){
  int i;

  for (i=0; i<N*N; i++){
    if (i%N == 0){
      myFile << endl;
    }
    myFile << matrix[i] << "\t";
  }
  myFile << endl;
}

// MAKE THESE MACROS
__host__ __device__ void writeMatrix(int *matrix, int x, int y, int value){
  matrix[x+(y*N)] = value;
}

__device__ int readMatrix(int *matrix, int x, int y){
  return matrix[x+(y*N)];
}

__global__ void calculateDistanceMatrix(int *matrix){
  // __shared__ int matrixCopy[N*N];

  int i, j, k;

  // for (i=0; i<N*N; i++){
  //   matrixCopy[i] = matrix[i];
  // }

  for (k=0; k<N; k++){
    for (j=0; j<N; j++){
      if ((j%BLOCK_SIZE) == blockIdx.x){
        for (i=0; i<N; i++){
          if ((i%THREAD_COUNT) == threadIdx.x){
            int minimum = min(readMatrix(matrix, i, j), readMatrix(matrix, i, k) + readMatrix(matrix, k, j));
            // EXPERIMENTAL
            writeMatrix(matrix, i, j, minimum);
            //writeMatrix(matrixCopy, i, j, minimum);
          }
        }
      }
    }
    // copy local matrix to device matrix
  }



  // divide rows up by block, and columns up by threads

  // since we are restricted to 8 blocks with 32 threads per block,
  // my strategy for a 100x100 matrix would be:

  // block 0 grabs row 0 and thread 0 would be responsible for the
  // 0th, 33rd, 66th, and 99th column, while process 2 would only be
  // responsible for the 2nd, 35th, and 68th column

  // At the same time, block 0 would be responsible for the 0th, 9th,
  // 18th, 27th, 36th, 45th, 54th, 63rd, 72nd, 81st, 90th, and 99th rows

  // As each row is completed, the matrix for that row will be updated, and
  // the block will move on to the next one. Once all blocks are finished,
  // that signals the completion of the kth step, and the blocks will update
  // the rows of the matrix in device memory that they were responsible for
  // and then synchronize blocks
  // By doing this, no race conditions should exist

  // This process will go from k = 0 to k = N-1 steps, and once all these
  // k steps are done, the function will be complete

}
