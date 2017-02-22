/*
Barrier implemented using tournament-style coding
*/

// Constraints: Number of processes must be a power of 2, e.g.
// 2,4,8,16,32,64,128,etc.

#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

void mybarrier(MPI_Comm);

// global debug bool
int verbose = 0;

int main(int argc, char * argv[]) {
	int rank;
	int size;

  int i;
  int sum = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int check = size;

	// check to make sure the number of processes is a power of 2
	if (rank == 0){
		if (size == 1){
			printf("ERROR: You must have more than one process!\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
			return 1;
		}
		while(check > 1){
			if (check % 2 == 0){
				check /= 2;
			} else {
				printf("ERROR: The number of processes must be a power of 2!\n");
				MPI_Abort(MPI_COMM_WORLD, 1);
				return 1;
			}
		}
	}

	if (rank == 0){
		printf("Starting task...\n");
	}
	// simple task, with barrier in the middle
  for (i = 0; i < 500; i++){
    sum ++;
  }
	// sleep(rank) means the tasks will call barrier at different times
	//  if main process (rank 0) waits for them as expected, it shouldn't
	//  print "Barrier complete!" until the highest rank process has called it
	sleep(rank);
	printf("Process %d calling barrier...\n", rank);
  mybarrier(MPI_COMM_WORLD);
	if (rank == 0){
		printf("Barrier complete!\n");
	}
  for (i = 0; i < 500; i++){
    sum ++;
  }
	if (rank == 0){
		printf("Task complete!\n");
	}

	if (verbose){
		printf("process %d arrived at finalize\n", rank);
	}
	MPI_Finalize();
	return 0;
}

void mybarrier(MPI_Comm comm){
	// MPI variables
  int rank;
  int size;

	int * data;

	// Loop variables
  int i;
	int a;

	int skip;

	int complete = 0;
	int currentCycle = 1;

	// Initialize MPI vars
  MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	// step 1, gathering
	while (!complete){
		skip = currentCycle * 2;

		// if currentCycle divides rank evenly, then it is a target
		if ((rank % currentCycle) == 0){
			// if skip divides rank evenly, then it needs to receive
			if ((rank % skip) == 0){
				MPI_Recv(data, 0, MPI_INT, rank + currentCycle, 99, comm, MPI_STATUS_IGNORE);
				if (verbose){
					printf("1: %d from %d\n", rank, rank + currentCycle);
				}
			// otherwise, it needs to send. Once sent, the process is done
			} else {
				if (verbose){
					printf("1: %d to %d\n", rank, rank - currentCycle);
				}
				MPI_Send(data, 0, MPI_INT, rank - currentCycle, 99, comm);
				complete = 1;
			}
		}

		currentCycle *= 2;

		// main process will never send, so this code will allow it to complete
		if (currentCycle >= size){
			complete = 1;
		}
	}

	complete = 0;
	currentCycle = size / 2;

	// step 2, scattering

	while (!complete){
		// if currentCycle is 1, then this is the last loop
		if (currentCycle == 1){
			complete = 1;
		}

		skip = currentCycle * 2;

		// if currentCycle divides rank evenly then it is a target
		if ((rank % currentCycle) == 0){
			// if skip divides rank evenly, then it needs to send
			if ((rank % skip) == 0){
				if (verbose){
					printf("2: %d to %d\n", rank, rank + currentCycle);
				}
				MPI_Send(data, 0, MPI_INT, rank + currentCycle, 99, comm);
			// otherwise, it needs to receive
			} else {
				if (verbose){
					printf("2: %d waiting for %d\n", rank, rank - currentCycle);
				}
				MPI_Recv(data, 0, MPI_INT, rank - currentCycle, 99, comm, MPI_STATUS_IGNORE);
				if (verbose){
					printf("2: %d from %d\n", rank, rank - currentCycle);
				}
			}
		}

		currentCycle /= 2;
	}
}
