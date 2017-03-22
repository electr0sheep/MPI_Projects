#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#define NUMBER_OF_CENTERS 16
#define DIMENSIONS 8

void assignPoints(short *, int);

int main(int argc, char * argv[]){
  int rank;
  int size;

  double data[12000*DIMENSIONS];
  double centers[NUMBER_OF_CENTERS*DIMENSIONS];
  short centerAssign[12000];

  MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  assignPoints(centerAssign, rank);
  printf("hello from %d\n", rank);

  MPI_Finalize();
  return 0;
}

void assignPoints(short * centerAssign, int rank){
  int a;
	short partialAssign[6000];
  // short * partialAssign;

  for (a=0; a<6000; a++){
    centerAssign[a] = a;
  }

	if (rank == 0){
    MPI_Recv(partialAssign, 6000, MPI_SHORT, 1, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (a=0; a<6000; a++){
      centerAssign[6000+a] = partialAssign[a];
    }
    printf("master %d\n", partialAssign[5]);
	} else {
    printf("slave %d\n", centerAssign[5]);
		MPI_Send(centerAssign, 6000, MPI_SHORT, 0, 15, MPI_COMM_WORLD);
	}
  printf("rank %d end of function\n", rank);
}
