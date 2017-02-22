#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char * argv[]) {
	int rank;
	int size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int A[1200][1200];	// A is the buffer used by the main process
	int B[2][1200];	// B is the buffer used by slave processes. There are two of them to support double buffering
	MPI_Request req[size-1][1200/size]; 	// each process has it's own request array, and each request is different
	MPI_Status status[size - 1];	// each process also has it's own status

	// these are the request and status objects used by the main processes
	MPI_Request mreq;
	MPI_Status mstatus;

	int flag;	// flag is relatively worthless but I use it for MPI_Test, which allows
						//  the program to print slave sums before the master process is completely
						//  finished



	//**************************************************************************//
	//														MASTER PROCESS																//
	//**************************************************************************//
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

				// also do this as part of the "x" loop
				// send to slave processes
				if ((x % size) > 0) {
					MPI_Isend(A[x], 1200, MPI_INT, (x % size), 99, MPI_COMM_WORLD, &mreq);
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
			}



	//**************************************************************************//
	//													SLAVE PROCESSES																	//
	//**************************************************************************//
	} else {

		// relativeRank allows us to easily access the proper array (otherwise,
		//  req[0] would just be wasted space)
		int relativeRank = rank - 1;

		// NOTE: if each process receives 200 requests, then there needs to be 201
		//  steps in this for loop, hence the abnormal <=
		for (int i = 0; i <= (1200/size); i++) {

			// this program uses double-buffering, so current buffer cycles between
			//  the two
			int currentBuffer = i % 2;
			// lastBuffer is used because we need to sum what was received on the
			//  last cycle, not the current one
			int lastBuffer = (i + 1) % 2;

	//**************************************************************************//
	//														FIRST TIME																		//
	//Process simply begins receiving data, and then moves on to the next step	//
	//**************************************************************************//
			if (i == 0) {
				MPI_Irecv(B[currentBuffer], 1200, MPI_INT, 0, 99, MPI_COMM_WORLD, &req[relativeRank][i]);

	//**************************************************************************//
	//															LAST TIME																		//
	//		Process waits for the last receive to finish, then sums and prints		//
	//**************************************************************************//
			} else if (i == (1200/size)) {

				int sum = 0;

				// wait for the last request to complete
				MPI_Wait(&req[relativeRank][i-1], &status[relativeRank]);

				// then sum it up
				for (int a = 0; a < 1200; a++) {
					sum += B[lastBuffer][a];
				}
				printf("SLAVE%d: %d\n", relativeRank, sum);

	//**************************************************************************//
	//													EVERY OTHER TIME																//
	// Process begins receiving data, then it waits for the previous receive to //
	//	complete, then sums and prints the data received previously							//
	//**************************************************************************//
			} else {
				int sum = 0;
				// receive data into the current buffer
				MPI_Irecv(B[currentBuffer], 1200, MPI_INT, 0, 99, MPI_COMM_WORLD, &req[relativeRank][i]);
				// wait for the previous request to finish
				MPI_Wait(&req[relativeRank][i-1], &status[relativeRank]);
				// sum the data contained in the previous buffer
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
