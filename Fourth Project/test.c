#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv []){

  int rank;
	int size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  int i, j, k;
  static int W0[6];

  if (rank == 0){
    for (i=0; i<6; i++){
      W0[i] = i;
    }
  }

  if (rank == 1){
    for (i=0; i<6; i++){
      printf("%d\n", W0[i]);
    }
    printf("ENDOFINITIAL\n");
  }

  if (rank == 0){
    MPI_Send(W0, 6, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(W0, 6, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (rank == 1){
    for (i=0; i<6; i++){
      printf("%d\n", W0[i]);
    }
  }

  MPI_Finalize();
  return 0;
}
