#include <mpi.h>
#include <stdio.h>

#define N 4
#define VERBOSE 1
#define DEBUG 0
#define CIRCULAR 0
#define LINE 1

int min(int, int);
void sendMatrix(int[]);
void calculateChunk(int[], int, int, int, int, int);
int readMatrix(int[], int, int, int);
void writeMatrix(int[], int, int, int, int);

int INT_MAX = 999999;

int main(int argc, char * argv []){

  int rank;
	int size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Make sure the square root of the number of processes
  //  is acceptable
  if (size != 1 && size != 4 && size != 9 && size != 16){
    if (rank == 0){
      printf("ERROR: You must use 1, 4, 9, or 16 processes\n");
    }
    MPI_Finalize();
    return 0;
  }

  int i, j, k;
  int W0[N*N];
  int W[N*N];

  int node1, node2;

  // Set up matrix to be a circular graph
  if (CIRCULAR){
    if (rank == 0){
      for (i=0; i<N; i++){
        for (j=0; j<N; j++){
          if (i == j+1 || j == i+1){
            writeMatrix(W0, i, j, N, 1);
          } else if ((i == N-1 && j == 0) || (i == 0 && j == N-1)){
            writeMatrix(W0, i, j, N, 1);
          } else{
            writeMatrix(W0, i, j, N, INT_MAX);
          }
        }
      }
    }
  // Set up matrix to be a line graph
  } else if (LINE){
    if (rank == 0){
      for (i=0; i<N; i++){
        for (j=0; j<N; j++){
          if (i == j+1 || j == i+1){
            writeMatrix(W0, i, j, N, 1);
          } else {
            writeMatrix(W0, i, j, N, INT_MAX);
          }
        }
      }
    }
  } else {
    if (rank == 0){
      printf("ERROR: Must set up either circular or line graph\n");
    }
    MPI_Finalize();
    return 0;
  }

  // Diagonal of graph must be 0
  if (rank == 0){
    for (i=0; i<N; i++){
      writeMatrix(W0, i, i, N, 0);
    }
  }

  sendMatrix(W0);

  // copy to W
  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      writeMatrix(W, i, j, N, readMatrix(W0, i, j, N));
    }
  }

  if (VERBOSE && rank == 0){
    printf("Initial Adjacency Matrix\n");
    printf("========================\n");
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        printf("%d\t", readMatrix(W0, i, j, N));
      }
      printf("\n");
    }
  }

  calculateChunk(W, 0, 0, N, rank, size);

  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      writeMatrix(W0, i, j, N, readMatrix(W, i, j, N));
    }
  }

  if (VERBOSE && rank == 0){
    printf("Resulting Distance Matrix\n");
    printf("=========================\n");
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        printf("%d\t", readMatrix(W0, i, j, N));
      }
      printf("\n");
    }
  }

  MPI_Finalize();
  return 0;
}

int min(int a, int b){
  if (a < b){
    return a;
  } else {
    return b;
  }
}

void sendMatrix(int W[N*N]){
  MPI_Bcast(W, N*N, MPI_INT, 0, MPI_COMM_WORLD);
}

void synchronizeMatrix(int matrix[], int x, int y, int width, int rank, int size){
  int subMatrix[width*width];
  int i, j;

  // copy the relevant data into subMatrix
  for (i=x; i<x+width; i++){
    for (j=y; j<y+width; j++){
      writeMatrix(subMatrix, i-x, j-y, width, readMatrix(matrix, i, j, N));
    }
  }

  if (DEBUG){
    printf("RANK %d SUBMATRIX\n", rank);
    printf("=================\n");
    for (i=0; i<width; i++){
      for (j=0; j<width; j++){
        printf("%d\t", readMatrix(subMatrix, i, j, width));
      }
      printf("\n");
    }
  }

  // // send the submatrix data back to master
  // if (rank == 0){
  //   for (i=0; i<size; i++){
  //     int row = (i*width)%
  //     MPI_Recv(subMatrix, width*width, MPI_INT, i, 0, MPI_COMM_WORLD);
  //     // copy the subMatrix data into master matrix
  //     writeMatrix(matrix, );
  //   }
  // } else {
  //   MPI_Send(subMatrix, width*width, MPI_INT, 0, 0, MPI_COMM_WORLD);
  // }
  //
  // // broadcast the new master matrix
  // sendMatrix(matrix);
}

int readMatrix(int matrix[], int x, int y, int width){
  return matrix[(x*width)+y];
}

void writeMatrix(int matrix[], int x, int y, int width, int value){
  matrix[(x*width)+y] = value;
}

void calculateChunk(int matrix[], int x, int y, int width, int rank, int size){
  int i, j, k;

  for (k=0; k<N; k++){
    for (i=x; i<x+width; i++){
      for (j=y; j<y+width; j++){
        int minimum = min(readMatrix(matrix, i, j, N), readMatrix(matrix, i, k, N) + readMatrix(matrix, k, j, N));
        writeMatrix(matrix, i, j, N, minimum);
      }
    }
    synchronizeMatrix(matrix, x, y, width, rank, size);
  }
}
