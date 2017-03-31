#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1200
#define VERBOSE 0
#define DEBUG 0
#define CIRCULAR 0
#define LINE 1

int min(int, int);
void sendMatrix(int[]);
void calculateChunk(int[], int, int, int, int, int);
int readMatrix(int[], int, int, int);
void writeMatrix(int[], int, int, int, int);
void synchronizeMatrix(int[], int[], int, int, int);
void updateMasterMatrix(int[], int, int, int, int);

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

void updateMasterMatrix(int masterMatrix[], int x, int y, int rank, int size){
  // x and y are the x and y coordinates of the start
  // of the process' submatrix
  int i, j, k;
  int currentSkip = 2;
  int width = N/size;
  int rankWidth = sqrt(size);
  int totalReceiveWidth = width*2;
  int secondaryMatrix[N*N];
  int rightRank = (rank+currentSkip)/2;
  int bottomRank = rank+rankWidth;
  int recvX, recvY;

  do{
    // first, all processes receive data from the process on the right
    for (i=0; i<rank; i++){
      if (rank%currentSkip == 0){
        recvX = x+width;
        recvY = y;
        MPI_Recv(secondaryMatrix, N*N, MPI_INT, rank+(currentSkip/2), i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (j=recvY; j<width; j++){
          for (k=recvX; k<width; k++){
            writeMatrix(masterMatrix, k, j, N, readMatrix(secondaryMatrix, k, j, N));
          }
        }
      } else {
        MPI_Send(masterMatrix, N*N, MPI_INT, rank-(currentSkip/2), i, MPI_COMM_WORLD);
      }
    }
    // then, receive data from the process below
    for (i=0; i<rank; i++){
      recvX = x;
      recvY = y+width;
      if (rank%(currentSkip*rankWidth) == 0){
        MPI_Recv(secondaryMatrix, N*N, MPI_INT, rank+((currentSkip/2)*width), i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (j=recvY; j<width*2; j++){
          for (k=recvX; k<width; k++){
            writeMatrix(masterMatrix, k, j, N, readMatrix(secondaryMatrix, k, j, N));
          }
        }
      } else {
        MPI_Send(masterMatrix, N*N, MPI_INT, rank-((currentSkip/2)*width), i, MPI_COMM_WORLD);
      }
    }
    currentSkip *=2;
    width *= 2;
  } while(width<=N);
}

// NOTE: I wasn't able to completely finish this, however I believe I could
// have accomplished the communication restriction using the following idea.
// First, send every other column to the left, and then every other row to the top
// Repeat this until the root process has received all data
// Here is a visual representation
//
// Proccess layout of matrix (each square represents a chunk of the overall
// matrix that each process is responsible for)
// |0 |1 |2 |3 |4 |5 |6 |7 |
// |8 |9 |10|11|12|13|14|15|
// |16|17|18|19|20|21|22|23|
// |24|25|26|27|28|29|30|31|
// |32|33|34|35|36|37|38|39|
// |40|41|42|43|44|45|46|47|
// |48|49|50|51|52|53|54|55|
// |56|57|58|59|60|61|62|63|
//
// STEP 1
// |0 |2 |4 |6 |       |0 |2 |4 |6 |
// |8 |10|12|14|       |16|18|20|22|
// |16|18|20|22|       |32|34|36|38|
// |24|26|28|30|       |48|50|52|54|
// |32|34|36|38|
// |40|42|44|46|
// |48|50|52|54|
// |56|58|60|62|
//
// STEP 2
// |0 |4 |             |0 |4 |
// |16|20|             |32|36|
// |32|36|
// |48|52|
//
// STEP 3
// |0 |                |0 |
// |32|
//
// Using this idea, the total number of communications should have been
// log(p)+2Nlog(p)
//
// Currently, my program uses log(p)+Nlog(p)+(p*N-1) as the total number of communications
//
// UPDATE: I believe I was close with updateMasterMatrix, but I ran out of time
