#include <iostream>

using namespace std;

#define N 10
#define BLOCK_SIZE 8
#define THREAD_COUNT 32

void initializeCircularGraph(int*);
void initializeLinearGraph(int*);
void writeMatrix(int*, int, int, int, int);
void printMatrix(int*);

int main(){
  int *matrix;
  int size = N * N * sizeof(int);

  matrix = (int *)malloc(size);

  initializeCircularGraph(matrix);

  // do cuda stuff

  cout << "                          ADJACENCY MATRIX" << endl;
  cout << "==============================================================================" << endl;
  printMatrix(matrix);

  // copy cuda stuff to host

  cout << "                            DISTANCE MATRIX" << endl;
  cout << "==============================================================================" << endl;
  printMatrix(matrix);

  free(matrix);

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
