#include <mpi.h>
#include <stdio.h>

#define N 5
#define VERBOSE 1
#define DEBUG 0
#define CIRCULAR 1
#define LINE 0

int min(int, int);
void sendChunks(int[N][N]);
void calculateChunk(int, int, int);

int main(int argc, char * argv []){

  int rank;
	int size;

  int INT_MAX = 10000;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Make sure the square root of the number of processes
  //  is acceptable
  if (size != 1 && size != 4 && size != 9){
    if (rank == 0){
      printf("ERROR: You must use 1, 4, or 9 processes\n");
    }
    MPI_Finalize();
    return 0;
  }

  int i, j, k;
  static int W0[N][N];
  static int W[N][N];

  int node1, node2;

  // Set up matrix to be a circular graph
  if (CIRCULAR){
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        if (i == j+1 || j == i+1){
          W0[i][j] = 1;
        } else if ((i == N-1 && j == 0) || (i == 0 && j == N-1)){
          W0[i][j] = 1;
        } else{
          W0[i][j] = INT_MAX;
        }
      }
    }
  // Set up matrix to be a line graph
  } else if (LINE){
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        if (i == j+1 || j == i+1){
          W0[i][j] = 1;
        } else {
          W0[i][j] = INT_MAX;
        }
      }
    }
  } else {
    MPI_Finalize();
    return 0;
  }

  // Diagonal of graph must be 0
  for (i=0; i<N; i++){
    W0[i][i] = 0;
  }

  // copy to W
  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      W[i][j] = W0[i][j];
    }
  }

  for (k=0; k<N; k++){
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        W[i][j] = min(W[i][j],W[i][k]+W[k][j]);
      }
    }
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        if (DEBUG){
          printf("Intermediary Matrix\n");
          printf("===================\n");
          for (i=0; i<N; i++){
            for (j=0; j<N; j++){
              printf("%d\t", W[i][j]);
            }
            printf("\n");
          }
        }
      }
    }
  }

  if (VERBOSE){
    printf("Initial Adjacency Matrix\n");
    printf("========================\n");
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        printf("%d\t", W0[i][j]);
      }
      printf("\n");
    }
  }

  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      W0[i][j] = W[i][j];
    }
  }

  if (VERBOSE){
    printf("Resulting Distance Matrix\n");
    printf("=========================\n");
    for (i=0; i<N; i++){
      for (j=0; j<N; j++){
        printf("%d\t", W0[i][j]);
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

void sendChunks(int W[N][N]){

}

void calculateChunk(int x, int y, int width){

}
