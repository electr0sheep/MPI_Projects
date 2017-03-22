#include <mpi.h>
#include <stdio.h>

int main(int argc, char * argv[]) {
	int rank;
	int size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Barrier(MPI_COMM_WORLD);
	printf("Hello from %d\n", rank);
	MPI_Barrier(MPI_COMM_WORLD);
	printf("Goodbye from %d\n", rank);
  MPI_Barrier(MPI_COMM_WORLD);
	printf("Hello 2 from %d\n", rank);
	MPI_Barrier(MPI_COMM_WORLD);
	printf("Goodbye 2 from %d\n", rank);
	MPI_Finalize();
	return 0;
}
