#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char * argv[]) {
	int rank;
	int size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// A is the buffer used by the main process
	int A[1200][1200];
	// B is the buffer used by slave processes
	//  there are two of them to support double buffering
	int B[2][1200];
	int current = 0;
	// each process has it's own request array, and each request is different
	MPI_Request req[size-1][1200/size];
	MPI_Status status[size - 1];
	// these are the request and status objects used by the main processes
	MPI_Request mreq;
	MPI_Status mstatus;
	// flag is relatively worthless but I use it for MPI_Test, which allows
	//  the program to print slave sums before the master process is completely
	//  finished
	int flag;

	if (rank == 0) {
		// if the number of processes can't evenly divide 1200 then
		//  we are going to have issues, so just abort
		if (1200 % size != 0) {
			printf("The number of process must exactly divide 1200\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
			return 1;
		}
		// generate numbers we use a non-random procedure for testing purposes
				for (int x = 0; x < 1200; x++) {
					for (int y = 0; y < 1200; y++) {
						A[x][y] = x + y;
					}
					// send to slave processes
					if (current > 0) {
						MPI_Isend(A[x], 1200, MPI_INT, current, 99, MPI_COMM_WORLD, &mreq);
						MPI_Test(&mreq, &flag, &mstatus);
					// have the master process take a turn summing a row
					// else gets executed first in a normal run
					} else {
						int sum = 0;
						for (int i = 0; i < 1200; i++) {
							sum += A[x][i];
						}
						printf("MASTER: %d\n", sum);
					}
					// current keeps track of which process should sum the current row
					// this code cycles current through all the processes
					current = (current + 1) % size;
				}
	} else {
		// relativeRank allows us to easily access the proper array (otherwise,
		// req[0] would just be wasted space)
		int relativeRank = rank - 1;
		for (int i = 0; i <= (1200/size); i++) {
			int currentBuffer = i % 2;
			// on the first call, simply start receiving data
			if (i == 0) {
				MPI_Irecv(B[currentBuffer], 1200, MPI_INT, 0, 99, MPI_COMM_WORLD, &req[relativeRank][i]);
			// on the last run, wait to receive the last row, then sum and print it
			} else if (i == (1200/size)) {
				int lastBuffer = (i + 1) % 2;
				int sum = 0;
				// wait for the last request to complete
				MPI_Wait(&req[relativeRank][i-1], &status[relativeRank]);
				// then sum it up
				for (int a = 0; a < 1200; a++) {
					sum += B[lastBuffer][a];
				}
				printf("SLAVE%d: %d\n", relativeRank, sum);
			// otherwise, begin receiving a new row, then wait for the last row to finish
			//  receiving, then sum it up and print it
			} else {
				int sum = 0;
				int lastBuffer = (i + 1) % 2;
				MPI_Irecv(B[currentBuffer], 1200, MPI_INT, 0, 99, MPI_COMM_WORLD, &req[relativeRank][i]);
				MPI_Wait(&req[relativeRank][i-1], &status[relativeRank]);
				for (int a = 0; a < 1200; a++) {
					sum += B[lastBuffer][a];
				}
				printf("SLAVE%d: %d\n", relativeRank, sum);
			}
		}
	}

	MPI_Finalize();

	return 0;
}
