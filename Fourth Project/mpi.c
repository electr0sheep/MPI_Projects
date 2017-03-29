#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4
#define VERBOSE 1
#define DEBUG 1
#define CIRCULAR 0
#define LINE 1

int min(int, int);
void sendMatrix(int[]);
void calculateChunk(int[], int, int, int, int, int);
int readMatrix(int[], int, int, int);
void writeMatrix(int[], int, int, int, int);
void synchronizeMatrix(int[], int[], int, int, int);

int INT_MAX = 999999;

int main(int argc, char * argv []){

  int rank;
	int size;

  int xChunk, yChunk, chunkWidth;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Make sure the square root of the number of processes is acceptable, and
  //  set parameters accordingly. Definitely not the ideal way to do this, but
  //  it was easy
  if (size == 1){
    chunkWidth = N;
    xChunk = 0;
    yChunk = 0;
  } else if (size == 4){
    chunkWidth = N/2;
    switch (rank){
      case 0:
      xChunk = 0;
      yChunk = 0;
      break;
      case 1:
      xChunk = 1;
      yChunk = 0;
      break;
      case 2:
      xChunk = 0;
      yChunk = 1;
      break;
      case 3:
      xChunk = 1;
      yChunk = 1;
      break;
    }
  } else if (size == 9){
    chunkWidth = N/3;
    switch (rank){
      case 0:
      xChunk = 0;
      yChunk = 0;
      break;
      case 1:
      xChunk = 1;
      yChunk = 0;
      break;
      case 2:
      xChunk = 2;
      yChunk = 0;
      break;
      case 3:
      xChunk = 0;
      yChunk = 1;
      break;
      case 4:
      xChunk = 1;
      yChunk = 1;
      break;
      case 5:
      xChunk = 2;
      yChunk = 1;
      break;
      case 6:
      xChunk = 0;
      yChunk = 2;
      break;
      case 7:
      xChunk = 1;
      yChunk = 2;
      break;
      case 8:
      xChunk = 2;
      yChunk = 2;
      break;
    }
  } else if (size == 16){
    chunkWidth = N/4;
    switch (rank){
      case 0:
      xChunk = 0;
      yChunk = 0;
      break;
      case 1:
      xChunk = 1;
      yChunk = 0;
      break;
      case 2:
      xChunk = 2;
      yChunk = 0;
      break;
      case 3:
      xChunk = 3;
      yChunk = 0;
      break;
      case 4:
      xChunk = 0;
      yChunk = 1;
      break;
      case 5:
      xChunk = 1;
      yChunk = 1;
      break;
      case 6:
      xChunk = 2;
      yChunk = 1;
      break;
      case 7:
      xChunk = 3;
      yChunk = 1;
      break;
      case 8:
      xChunk = 0;
      yChunk = 2;
      break;
      case 9:
      xChunk = 1;
      yChunk = 2;
      break;
      case 10:
      xChunk = 2;
      yChunk = 2;
      break;
      case 11:
      xChunk = 3;
      yChunk = 2;
      break;
      case 12:
      xChunk = 0;
      yChunk = 3;
      break;
      case 13:
      xChunk = 1;
      yChunk = 3;
      break;
      case 14:
      xChunk = 2;
      yChunk = 3;
      break;
      case 15:
      xChunk = 3;
      yChunk = 3;
      break;
    }
  } else {
    if (rank == 0){
      printf("ERROR: You must use 1, 4, 9, or 16 processes\n");
    }
    MPI_Finalize();
    return 0;
  }

  xChunk *= chunkWidth;
  yChunk *= chunkWidth;

  int i, j, k;
  int W0[N*N];

  // Set up matrix to be a circular graph
  if (CIRCULAR){
    if (rank == 0){
      for (j=0; j<N; j++){
        for (i=0; i<N; i++){
          if (j == i+1 || i == j+1){
            writeMatrix(W0, i, j, N, 1);
          } else if ((j == N-1 && i == 0) || (j == 0 && i == N-1)){
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
      for (j=0; j<N; j++){
        for (i=0; i<N; i++){
          if (j == i+1 || i == j+1){
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

  calculateChunk(W0, xChunk, yChunk, chunkWidth, rank, size);

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

// The purpose of this function is to update the master matrix as the distance
//  matrix is calculated, and spread that information to all the processes
void synchronizeMatrix(int matrix[], int W[], int width, int rank, int size){
  int i, j, k;
  int numberOfColumns;
  int currentColumn = 0;
  int currentRow = 0;

  if (size == 1){
    numberOfColumns = 1;
  } else if (size == 4){
    numberOfColumns = 2;
  } else if (size == 9){
    numberOfColumns = 3;
  } else if (size == 16){
    numberOfColumns = 4;
  }

  // send the submatrix data back to master
  if (rank == 0){
    // first copy the master data into primary matrix
    for (j=0; j<width; j++){
      for (i=0; i<width; i++){
        writeMatrix(matrix, i, j, width*numberOfColumns, readMatrix(W, i, j, width));
      }
    }
    // then receive the other processes' data
    for (i=1; i<size; i++){
      currentColumn++;
      if (currentColumn >= numberOfColumns){
        currentColumn = 0;
        currentRow++;
      }
      MPI_Recv(W, width*width, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // copy the subMatrix data into master matrix
      for (j=0; j<width; j++){
        for (k=0; k<width; k++){
          writeMatrix(matrix, (currentColumn*width)+k, (currentRow*width)+j, width*numberOfColumns, readMatrix(W, k, j, width));
        }
      }
    }
  } else {
    MPI_Send(W, width*width, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  // broadcast the new master matrix
  sendMatrix(matrix);
}

int readMatrix(int matrix[], int x, int y, int width){
  return matrix[x+(y*width)];
}

void writeMatrix(int matrix[], int x, int y, int width, int value){
  matrix[x+(y*width)] = value;
}

void calculateChunk(int matrix[], int x, int y, int width, int rank, int size){
  int i, j, k;

  int W[width*width];

  for (k=0; k<N; k++){
    for (j=y; j<y+width; j++){
      for (i=x; i<x+width; i++){
        int minimum = min(readMatrix(matrix, i, j, N), readMatrix(matrix, i, k, N) + readMatrix(matrix, k, j, N));
        writeMatrix(W, i-x, j-y, width, minimum);
      }
    }
    synchronizeMatrix(matrix, W, width, rank, size);
  }
}
