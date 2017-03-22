#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#define NUMBER_OF_CENTERS 16
#define DIMENSIONS 8

void assignPoints(double *, double *, short *, int, int, int, int);
// void assignPoints(double *, double *, short *, int, int, int, int, short *);
double distance(int, double *, double *);

int main(int argc, char * argv[]){
  int a, b;

  int rank;
  int size;

  double data[12000*DIMENSIONS];
  double centers[NUMBER_OF_CENTERS*DIMENSIONS];
  short centerAssign[12000];

  MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  short partialAssign[12000/size];

  if (rank == 0){
    srand(25);
    for (a=0; a<12000*DIMENSIONS; a++){
      data[a] = rand();
    }
    for (a=0; a<NUMBER_OF_CENTERS; a++){
      for (b=0; b<DIMENSIONS; b++){
        centers[a*DIMENSIONS+b] = rand();
      }
    }
    if (size > 1){
      MPI_Send(data, 12000*DIMENSIONS, MPI_DOUBLE, 1, 5, MPI_COMM_WORLD);
      MPI_Send(centers, NUMBER_OF_CENTERS*DIMENSIONS, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(data, 12000*DIMENSIONS, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(centers, NUMBER_OF_CENTERS*DIMENSIONS, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  printf("process %d calling assign points\n", rank);
  // assignPoints(data, centers, centerAssign, rank*6000, 6000, rank, size, partialAssign);
  assignPoints(data, centers, centerAssign, rank*6000, 6000, rank, size);

  MPI_Finalize();
  return 0;
}

void assignPoints(double * data, double * centers, short * centerAssign, int start, int size, int rank, int numProcesses){
// void assignPoints(double * data, double * centers, short * centerAssign, int start, int size, int rank, int numProcesses, short * partialAssign){
	int a, b;
	short partialAssign[12000];
	double currentClosestDistance, currentDistance;
	short currentClosestCenter;

	// all processes calculate
	for (a=0; a<size; a++){
		currentClosestDistance = DBL_MAX;
		// find center closest to current point
		for (b=0; b<NUMBER_OF_CENTERS; b++){
			currentDistance = distance(DIMENSIONS, &data[rank*size*DIMENSIONS], &centers[b*DIMENSIONS]);
			if (currentDistance < currentClosestDistance){
				currentClosestDistance = currentDistance;
				currentClosestCenter = b;
			}
		}
		centerAssign[a] = currentClosestCenter;
	}

	printf("process %d completed first part\n", rank);

	if (rank == 0){
		// first assign it's own findings to complete assignment set
		// then, receive data from other processes
		// for (a=1; a<numProcesses; a++){
		// 	MPI_Recv(partialAssign, size, MPI_DOUBLE, a, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// 	printf("no seg fault yet...\n");
		// 	printf("%d\n", partialAssign[0]);
		// 	printf("there was a seg fault...or was there?\n");
		// 	for (b=0; b<size; b++){
		// 		centerAssign[a*size+b] = partialAssign[b];
		// 	}
		// }
    MPI_Recv(partialAssign, size, MPI_SHORT, 1, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("master completed second part\n");
    printf("master size of partial assign %d\n", size);
    printf("master address of first sent el\t%p\n", &partialAssign);
    printf("no seg fault yet...\n");
    printf("rank is %d\n", rank);
		printf("%d\n", partialAssign[0]);
		printf("there was a seg fault...or was there?\n");
		for (b=0; b<size; b++){
      printf("1\n");
			centerAssign[a*size+b] = partialAssign[b];
		}
	}
	// All other processes
	else {
		// send results to master
    printf("slave address of first sent el %p\n", &centerAssign);
		MPI_Send(centerAssign, size, MPI_SHORT, 0, 15, MPI_COMM_WORLD);
		printf("%d completed second part\n", rank);
	}
	// impose a barrier so the pointers don't go out of scope until master is ready
	MPI_Barrier(MPI_COMM_WORLD);
	printf("%d has bassed the barrier\n", rank);
}

double distance(int dim, double * datum1, double * datum2){
  int i;
  double dist = 0.0;

  for (i=0; i<dim; i++){
    dist += pow(datum1[i]-datum2[i], 2);
  }

  return sqrt(dist);
}
