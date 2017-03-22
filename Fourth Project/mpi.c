#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8
#define SEED 10

int min(int, int);

int main(int argc, char * argv []){

  int rank;
	int size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  int i, j, k;
  static int W0[N][N];
  static int W[N][N];

  int node1, node2;

  // set up random seed
  srand(SEED);

  // initialize matrix
  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      W0[i][j] = 0;
    }
  }

  // There are N nodes, so randomly set up half of these nodes to be linked
  for (i=0; i<N/2; i++){
    node1 = rand() % N;
    node2 = rand() % N;
    // no diagonal entries can be 1
    while (node1 == node2){
      node1 = rand() % N;
      node2 = rand() % N;
    }
    W0[node1][node2] = 1;
    W0[node2][node1] = 1;
  }

  for (k=0; k<N-1; k++){
    for (i=0; i<N-1; i++){
      for (j=0; j<N-1; j++){
        W[i][j] = min(W0[i][j],W0[i][k]+W0[k][j]);
      }
    }
  }

  printf("Initial Matrix\n");
  printf("==============\n");
  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      printf("%d ", W0[i][j]);
    }
    printf("\n");
  }

  for (i=0; i<N-1; i++){
    for (j=0; j<N-1; j++){
      W0[i][j] = W[i][j];
    }
  }

  printf("Resulting Matrix\n");
  printf("================\n");
  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      printf("%d ", W0[i][j]);
    }
    printf("\n");
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
