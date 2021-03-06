/*****************************************************************************************/
/* https://vtechworks.lib.vt.edu/bitstream/handle/10919/19873/TR_GPU_synchronization.pdf */
/*****************************************************************************************/

/***************************************************************************/
/* THIS PROGRAM IMPLEMENTS THE SIMPLE GPU SYNC FUNCTION IN THE ABOVE PAPER */
/***************************************************************************/

// 7314 clock cycles for cpu bound sync (10 runs)
// 7193 clcok cycles for gpu sync (10 runs)

#include <iostream>
#include <fstream>

using namespace std;

#define N 64
#define BLOCK_SIZE 8
#define THREAD_COUNT 32
#define FILE_NAME "output.txt"

#define WRITE_MATRIX(M, X, Y, V) \
  M[X+(Y*N)] = V;

#define READ_MATRIX(M, X, Y, R) \
  R = M[X+(Y*N)];

void printMatrix(int*, ofstream&);
bool checkCircularResults(int*);
bool checkLinearResults(int*);
__global__ void calculateDistanceMatrix(int*);
__global__ void initializeCircularGraph(int*);
__global__ void initializeLinearGraph(int*);
// GPU simple synchronization function
__device__ void __gpu_sync(int);

// the mutex variable
__device__ int d_mutex = 0;

int main(){
  // first, make sure N is even and in range
  if (N%2 != 0){
    cout << "ERROR: N must be even. Change N and recompile!" << endl;
    return 1;
  } else if (N > 128){
    cout << "ERROR: The max size of N is 128. Change N and recompile!" << endl;
    return 1;
  }

  // variables
  int *matrix;
  int *d_matrix;
  int size = N * N * sizeof(int);
  ofstream myFile;

  myFile.open(FILE_NAME);

  // Allocate space for device copy of matrix
  cudaMalloc((void **)&d_matrix, size);

  // Allocate space for host copy of matrix
  matrix = (int *)malloc(size);

  // Copy matrix to device and initialize
  cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice);
  initializeCircularGraph<<<BLOCK_SIZE,THREAD_COUNT>>>(d_matrix);

  // Copy initialized matrix to host
  cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);

  // do cuda stuff
  // NOTE: CUDA does not have any way to synchronize blocks. Because we need
  // all blocks to by synced after each K step, the way to accomplish this is
  // to end the kernel whenever block synchronization is required.
  calculateDistanceMatrix<<<BLOCK_SIZE,THREAD_COUNT>>>(d_matrix);

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

  cout << "Number of clock ticks: " << clock() << endl;

  return 0;
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
        cout << "Matrix check failed!" << endl;
        cout << "These matrix coordinates failed:" << endl;
        cout << "X: " << x << " Y: " << y << endl;
        cout << "Found " << n << " where " << number << " was expected" << endl;
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
  cout << "Matrix check passed!" << endl;
  return true;
}

__global__ void calculateDistanceMatrix(int *matrix){
  int i, j, k, num1, num2, num3;
  int currentGoalVal = 0;

  for (k=0; k<N; k++){
    for (j=0; j<N; j++){
      if ((j%BLOCK_SIZE) == blockIdx.x){
        for (i=0; i<N; i++){
          if ((i%THREAD_COUNT) == threadIdx.x){
            READ_MATRIX(matrix, i, j, num1);
            READ_MATRIX(matrix, i, k, num2);
            READ_MATRIX(matrix, k, j, num3);
            int minimum = min(num1, num2 + num3);
            WRITE_MATRIX(matrix, i, j, minimum);
          }
        }
      }
    }
    currentGoalVal += BLOCK_SIZE;
    __gpu_sync(currentGoalVal);
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
  // that signals the completion of the kth step, and the kernel will terminate

  // By terminating the kernel after every kth step, this ensures that all
  // blocks remain synchronized

}

__global__ void initializeCircularGraph(int *matrix){
  int i, j;

  for (j=0; j<N; j++){
    if ((j%BLOCK_SIZE) == blockIdx.x){
      for (i=0; i<N; i++){
        if ((i%THREAD_COUNT) == threadIdx.x){
          // diagonal is zero
          if (i == j){
            WRITE_MATRIX(matrix, i, j, 0);
          // makes graph linear
          } else if (i == j+1 || j == i+1){
            WRITE_MATRIX(matrix, i, j, 1);
          // makes graph circular
          } else if((i == 0 && j == N-1) || (j == 0 && i == N - 1)){
            WRITE_MATRIX(matrix, i, j, 1);
          // every other node isn't connected
          } else {
            WRITE_MATRIX(matrix, i, j, 999999)
          }
        }
      }
    }
  }
}

__global__ void initializeLinearGraph(int *matrix){
  int i, j;

  for (j=0; j<N; j++){
    if ((j%BLOCK_SIZE) == blockIdx.x){
      for (i=0; i<N; i++){
        if ((i%THREAD_COUNT) == threadIdx.x){
          // diagonal is zero
          if (i == j){
            WRITE_MATRIX(matrix, i, j, 0);
          // makes graph linear
          } else if (i == j+1 || j == i+1){
            WRITE_MATRIX(matrix, i, j, 1);
          // every other node isn't connected
          } else {
            WRITE_MATRIX(matrix, i, j, 999999)
          }
        }
      }
    }
  }
}

__device__ void __gpu_sync(int goalVal){
  // only thread 0 is used for synchronization
  if (threadIdx.x == 0){
    atomicAdd(&d_mutex, 1);

    printf("Block %d looking for %d\n", blockIdx.x, goalVal);

    // < ensures that no block will get stuck in loop if another block executes
    // __gpu_sync before it has a chance to check the value
    while(d_mutex < goalVal){
      // ...
    }

    printf("Block %d is no longer stuck in loop\n", blockIdx.x);
  }
  __syncthreads();
}
