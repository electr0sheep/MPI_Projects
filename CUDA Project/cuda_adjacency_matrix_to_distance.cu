#include <iostream>
#include <fstream>

using namespace std;

#define N 128
#define BLOCK_SIZE 8
#define THREAD_COUNT 32
#define FILE_NAME "output.txt"

#define WRITE_MATRIX(M, X, Y, V) \
  M[X+(Y*N)] = V;

#define READ_MATRIX(M, X, Y, R) \
  R = M[X+(Y*N)];

void initializeCircularGraph(int*);
void initializeLinearGraph(int*);
void printMatrix(int*, ofstream&);
bool checkCircularResults(int*);
bool checkLinearResults(int*);
__global__ void calculateDistanceMatrix(int*, int);

int main(){
  // first, make sure N is even
  if (N%2 != 0){
    cout << "ERROR: N must be even. Change N and recompile!" << endl;
    return 1;
  } else if (N > 128){
    cout << "ERROR: The max size of N is 128. Change N and recompile!" << endl;
    return 1;
  }
  int *matrix;
  int *d_matrix;
  int k;
  int size = N * N * sizeof(int);
  ofstream myFile;

  myFile.open(FILE_NAME);

  // Allocate space for device copy of matrix
  cudaMalloc((void **)&d_matrix, size);

  // Allocate space for host copy of matrix and initialize
  matrix = (int *)malloc(size); initializeCircularGraph(matrix);

  // Copy matrix to device
  cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice);

  // do cuda stuff
  // NOTE: CUDA does not have any way to synchronize blocks. Because we need
  // all blocks to by synced after each K step, the way to accomplish this is
  // to end the kernel whenever block synchronization is required.
  for (k=0; k<N; k++){
    calculateDistanceMatrix<<<BLOCK_SIZE,THREAD_COUNT>>>(d_matrix, k);
  }

  myFile << "                          ADJACENCY MATRIX" << endl;
  myFile << "==============================================================================" << endl;
  printMatrix(matrix, myFile);

  // Copy results back to host
  cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);

  myFile << "                            DISTANCE MATRIX" << endl;
  myFile << "==============================================================================" << endl;
  printMatrix(matrix, myFile);

  myFile << "RESULT OF MATRIX CHECK: " << checkCircularResults(matrix) << endl;

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
        WRITE_MATRIX(ar, i, j, 1);
      } else if ((j == N-1 && i == 0) || (j == 0 && i == N-1)){
        WRITE_MATRIX(ar, i, j, 1);
      } else{
        WRITE_MATRIX(ar, i, j, 999999);
      }
    }
  }

  for (i=0; i<N; i++){
    WRITE_MATRIX(ar, i, i, 0);
  }
}

void initializeLinearGraph(int *ar){
  int i, j;

  for (j=0; j<N; j++){
    for (i=0; i<N; i++){
      if (j == i+1 || i == j+1){
        WRITE_MATRIX(ar, i, j, 1);
      } else {
        WRITE_MATRIX(ar, i, j, 999999);
      }
    }
  }

  for (i=0; i<N; i++){
    WRITE_MATRIX(ar, i, i, 0);
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

bool checkCircularResults(int *matrix){
  int x, y, number, n;
  bool increment = true;

  number = 0;

  for (y=0; y<N; y++){
    for (x=0; x<N; x++){
      READ_MATRIX(matrix, x, y, n);
      if (number != n){
        cout << "X: " << x << " Y: " << y << endl;
        cout << "number: " << number << " matrix: " << n << endl;
        cout << "increment: " << increment << endl;
        return false;
      }
      if (number == N/2){
        increment = false;
      } else if (number == 0){
        increment = true;
      }
      if (increment == true){
        number ++;
      } else {
        number --;
      }
    }
    if (y < N/2){
      number++;
    } else {
      number--;
    }
  }
  return true;
}

__global__ void calculateDistanceMatrix(int *matrix, int k){
  // __shared__ int matrixCopy[N*N];

  int i, j, num1, num2, num3;

  // for (i=0; i<N*N; i++){
  //   matrixCopy[i] = matrix[i];
  // }

  for (j=0; j<N; j++){
    if ((j%BLOCK_SIZE) == blockIdx.x){
      for (i=0; i<N; i++){
        if ((i%THREAD_COUNT) == threadIdx.x){
          READ_MATRIX(matrix, i, j, num1);
          READ_MATRIX(matrix, i, k, num2);
          READ_MATRIX(matrix, k, j, num3);
          int minimum = min(num1, num2 + num3);
          // EXPERIMENTAL
          WRITE_MATRIX(matrix, i, j, minimum);
          //writeMatrix(matrixCopy, i, j, minimum);
        }
      }
    }
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
