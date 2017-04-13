#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 50000
#define INT_MAX 9999999
#define MAX_PROCESSES 32
#define VERBOSE 0

void initializeMatrix(double[], int);
void force_calc(double[], double[], int, int);
void send_force_data_to_root(double[], int, int);
void print_particle_data(double[], double[]);

int main(int argc, char * argv []){
  int rank, size;
  double start, end;

  MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 1 && size != 2 && size != 4  && size != 8 && size != 16 && size != 32){
    if (rank == 0){
      printf("ERROR: Must use 1, 2, 4, 8, 16, or 32 processes!\n");
    }
    MPI_Finalize();
    return 0;
  }

  double positions[N], forces[N];
  int x;
  int i;

  if (rank == 0){
    initializeMatrix(positions, N);
  }

  MPI_Bcast(positions, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // get the start time
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  force_calc(positions, forces, rank, size);

  send_force_data_to_root(forces, rank, size);

  // get the finish time
  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();

  if (rank == 0){
    printf("Runtime = %f\n", end-start);
    if (VERBOSE){
      print_particle_data(positions, forces);
    }
  }

  MPI_Finalize();
  return 0;
}

void initializeMatrix(double matrix[], int width){
  int x;

  for (x=0; x<width; x++){
    matrix[x] = x;
  }
}

void force_calc(double x[], double F[], int rank, int size){
  int i, j;
  int currentProcess = 0;
  int increment = 1;
  double tmp;

  for (i=0; i<N; i++){
    F[i] = 0.0;
  }

  for (i=1; i<N; i++){
    if (currentProcess == rank){
      for (j=0; j<i; j++){
        tmp = 1.0/((x[i]-x[j])*(x[i]-x[j]));
        F[i] += tmp;
        F[j] -= tmp;
      }
    }
    currentProcess += increment;
    if (currentProcess == size){
      increment = -1;
      currentProcess--;
    } else if (currentProcess == -1){
      increment = 1;
      currentProcess++;
    }
  }
}

void send_force_data_to_root(double F[], int rank, int size){
  int i;
  double tmp[N];
  int step = 1;
  int recv;

  for (step=1; step <= size/2; step*=2){
    recv=step*2;
    if (rank%step == 0){
      if (rank%recv == 0){
        MPI_Recv(tmp, N, MPI_DOUBLE, rank+step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (i=0; i<N; i++){
          F[i] += tmp[i];
        }
      } else {
        MPI_Send(F, N, MPI_DOUBLE, rank-step, 0, MPI_COMM_WORLD);
      }
    }
  }
}

void print_particle_data(double positions[], double forces[]){
  int x;

  printf("                               POSITION ARRAY\n");
  printf("================================================================================\n");
  for (x=0; x<N; x++){
    printf("%f\t", positions[x]);
  }

  printf("                                FORCES ARRAY\n");
  printf("================================================================================\n");
  for (x=0; x<N; x++){
    printf("%f\t", forces[x]);
  }
  printf("\n");
}
