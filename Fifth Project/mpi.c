#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4

void initializeMatrix(double[], int);
double readMatrix(double[], int, int, int);
void writeMatrix(double[], int, int, int, double);
void printMatrix(double[], int);
void force_calc(double[], double[]);

int main(int argc, char * argv []){
  int rank, size;

  MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  double positions[N], forces[N*N];
  int x;

  initializeMatrix(positions, N);

  force_calc(positions, forces);

  printf("                               POSITION MATRIX\n");
  printf("================================================================================\n");
  for (x=0; x<N; x++){
    printf("%f\t", positions[x]);
  }
  printf("\n\n");
  printf("                                FORCES MATRIX\n");
  printf("================================================================================\n");

  printMatrix(forces, N);

  MPI_Finalize();
  return 0;
}

void initializeMatrix(double matrix[], int width){
  // int x;
  //
  // for (x=0; x<width; x++){
  //   matrix[x] = x;
  // }
  matrix[0]=3;
  matrix[1]=1;
  matrix[2]=2;
  matrix[3]=4;
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

void force_calc(double * x, double * F){
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
