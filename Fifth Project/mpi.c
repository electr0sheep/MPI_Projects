#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100
#define INT_MAX 9999999
#define MAX_PROCESSES 16
#define VERBOSE 0

void initializeMatrix(double[], int);
double readMatrix(double[], int, int, int);
void writeMatrix(double[], int, int, int, double);
void printMatrix(double[], int);
void force_calc_matrix(double[], double[]);
void assign_processes(int[], int, int);
void force_calc(double[], double[], int[], int);
void send_force_data_to_root(double[], int, int);


int main(int argc, char * argv []){
  int rank, size;

  MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  double positions[N], forces[N], forces_matrix[N*N];
  int x;
  int i;
  int row_assignments[N];

  assign_processes(row_assignments, N-1, size);

  if (rank == 0){
    initializeMatrix(positions, N);
  }

  MPI_Bcast(positions, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  force_calc(positions, forces, row_assignments, rank);

  send_force_data_to_root(forces, rank, size);

  if (rank == 0 && VERBOSE){
    force_calc_matrix(positions, forces_matrix);
    printf("                               POSITION MATRIX\n");
    printf("================================================================================\n");
    for (x=0; x<N; x++){
      printf("%f\t", positions[x]);
    }
    printf("\n\n");
    printf("                                FORCES MATRIX\n");
    printf("================================================================================\n");

    printMatrix(forces_matrix, N);

    printf("                                FORCES ARRAY\n");
    printf("================================================================================\n");
    for (x=0; x<N; x++){
      printf("%f\t", forces[x]);
    }
    printf("\n");
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

double readMatrix(double matrix[], int x, int y, int width){
  return matrix[x+(y*width)];
}

void writeMatrix(double matrix[], int x, int y, int width, double value){
  matrix[x+(y*width)] = value;
}

void printMatrix(double matrix[], int width){
  int x, y;

  for (y=0; y<width; y++){
    for (x=0; x<width; x++){
      printf("%f\t", readMatrix(matrix, x, y, width));
    }
    printf("\n");
  }
}

void force_calc_matrix(double * x, double * F){
  int i, j;
  double tmp;

  for (i=0; i<N; i++){
    F[i] = 0.0;
  }

  for (i=0; i<N; i++){
    for (j=0; j<i; j++){
      tmp = 1.0/((x[i]-x[j])*(x[i]-x[j]));
      if (x[i] < x[j]){
        writeMatrix(F, j, i, N, tmp);
        writeMatrix(F, i, j, N, -tmp);
      } else if (x[i] > x[j]) {
        writeMatrix(F, j, i, N, -tmp);
        writeMatrix(F, i, j, N, tmp);
      } else {
        printf("Two particles have occupied the same space, exiting...\n");
        exit(0);
      }
    }
  }
}

// NOTE: this algorithm could be simplified in this special case by
// simply assigning the row to rank 0 to rank N, then from rank N back
// to rank 0, then from rank 0 back to rank N and so on
void assign_processes(int row_assignments[], int size_of_bottom_row, int size){
  int size_of_current_row = size_of_bottom_row;
  int rank_load[MAX_PROCESSES];
  int least_load, least_load_rank;
  int i;
  int currentRow = N-1;

  // initialize rank_load
  for (i=0; i<MAX_PROCESSES; i++){
    rank_load[i] = 0;
  }

  while (size_of_current_row > 0){
    least_load = INT_MAX;
    // first, select the process with the least load
    for (i=0; i<size; i++){
      if (rank_load[i] < least_load){
        least_load = rank_load[i];
        least_load_rank = i;
      }
    }

    // then, assign the max load to the selected process
    // NOTE: the max load is always the current load in this case
    row_assignments[currentRow] = least_load_rank;
    rank_load[least_load_rank] += size_of_current_row;
    size_of_current_row--;
    currentRow--;
  }
}

void force_calc(double x[], double F[], int assignments[], int rank){
  int i, j;
  double tmp;

  for (i=0; i<N; i++){
    F[i] = 0.0;
  }

  for (i=0; i<N; i++){
    if (assignments[i] == rank){
      for (j=0; j<i; j++){
        tmp = 1.0/((x[i]-x[j])*(x[i]-x[j]));
        F[i] += tmp;
        F[j] -= tmp;
      }
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
