/*
Parallelized K-Mean program
We are using 8 dimensional points
The number of points shall be 12,000
*/

/*
Instead of using two-dimensional arrays to store points,
just use 1 dimensional array. I.E. If ar[0] is the start of the point,
ar[0] contains the 1st dimension point, a[1] the second, and so on
*/

// Constraints: Number of processes must be a power of 2, e.g.
// 2,4,8,16,32,64,128,etc.

// Parallelize assigning points to centers
// parallelize centering centers to data points

#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>

#define DIMENSIONS 8
#define NUMBER_OF_POINTS 12000
#define NUMBER_OF_CENTERS 16
#define MAX_RANDOM_NUMBER 1000
#define THRESHOLD 0.001
#define VERBOSE 1
#define PRINT_TIME_SPENT 1

void initialize_data(int, int, double *, double *);
void kmean(int, int, double *, short, short *, double *, int *, int, int);
double distance(int, double *, double *);
void recenterCenters(double *, double *, short *, int, int, int, int);
void assignPoints(double *, double *, short *, int, int, int, int);


int main(int argc, char * argv[]) {
	int rank;
	int size;

  int i;
  int k;
	int a;
  double data[NUMBER_OF_POINTS*DIMENSIONS];
  short cluster_assign[NUMBER_OF_POINTS];
  double cluster_centers[NUMBER_OF_CENTERS*DIMENSIONS];
  int cluster_size[NUMBER_OF_CENTERS];

	time_t time1 = time(NULL);
	time_t time2;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Request req;

	initialize_data(rank, size, data, cluster_centers);

	if (rank == 0){
		// assign each center to a data point
		for (i=0; i<NUMBER_OF_CENTERS; i++){
			for (k=0;k<DIMENSIONS; k++){
				cluster_centers[i*DIMENSIONS+k] = data[i*DIMENSIONS+k];
			}
		}
		// send the centers to the other processes
		for (i=0; i<size; i++){
			MPI_Send(cluster_centers, NUMBER_OF_CENTERS*DIMENSIONS, MPI_DOUBLE, i, 2, MPI_COMM_WORLD);
		}
	}
	// other processes need to receive center data
	else {
		MPI_Recv(cluster_centers, NUMBER_OF_CENTERS*DIMENSIONS, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

  kmean(DIMENSIONS, NUMBER_OF_POINTS, data, NUMBER_OF_CENTERS, cluster_assign, cluster_centers, cluster_size, size, rank);

	// NO PROCESS CAN MOVE PASSED THIS POINT UNTIL ALL PROCESSES ARE DONE
	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0){
		time2 = time(NULL);
		if (PRINT_TIME_SPENT){
			printf("COMPUTATION TOOK %f SECONDS\n", difftime(time2, time1));
		}
	}

	MPI_Finalize();
	return 0;
}

void initialize_data(int rank, int size, double * point_data, double * center_data){
	int a;
	int b;
	int c;
	int selectedProcess = 0;
	int currentPoint = 0;
	double partialPointData[(NUMBER_OF_POINTS/size)*DIMENSIONS];
	double partialCenterData[(NUMBER_OF_CENTERS/size)*DIMENSIONS];
	int partialSize = NUMBER_OF_POINTS/size;

	srand(rank*50);

	// randomize data
	for (a=0; a<NUMBER_OF_POINTS; a++){
		selectedProcess = a % size;
		if (rank == selectedProcess){
			for (b=0; b<DIMENSIONS; b++){
				partialPointData[currentPoint*DIMENSIONS+b] = rand() % MAX_RANDOM_NUMBER;
			}
			currentPoint++;
		}
	}

	if (rank == 0){
		// master first assigns it's own data points
		for (a=0; a<currentPoint; a++){
			for (b=0; b<DIMENSIONS; b++){
				point_data[a*DIMENSIONS+b] = partialPointData[a*DIMENSIONS+b];
			}
		}
		// send data back to master
		for (a=1; a<size; a++){
			MPI_Recv(partialPointData, partialSize*DIMENSIONS, MPI_DOUBLE, a, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			for (b=0; b<partialSize; b++){
				for (c=0; c<DIMENSIONS; c++){
					point_data[currentPoint*DIMENSIONS+c] = partialPointData[b*DIMENSIONS+c];
				}
				currentPoint++;
			}
		}
		for (a=1; a<size; a++){
			MPI_Send(point_data, NUMBER_OF_POINTS*DIMENSIONS, MPI_DOUBLE, a, 1, MPI_COMM_WORLD);
		}
	}
	// all other processes receive data from master and assign to point_data
	else {
		MPI_Send(partialPointData, partialSize*DIMENSIONS, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(point_data, NUMBER_OF_POINTS*DIMENSIONS, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}

void kmean(int dim, int n, double * data, short k, short * cluster_assign, double * cluster_centers, int * cluster_size, int size, int rank){
  int a, b, c;
  double largest_change = DBL_MAX;
  double initial_centers[k*dim];
	double averages[k*dim];
	int currentStep = 0;
	int allCentersAssigned = 0;
	short initial_cluster_assign[n];
	int allPointsAssignedSameCenter = 0;
	int keepGoing[1] = {1};
	int recenterAgain[1] = {1};
	double partialClusterCenters[(NUMBER_OF_CENTERS/size)*DIMENSIONS];
	double partialClusterSize[NUMBER_OF_CENTERS];
	int partialSize = NUMBER_OF_POINTS/size;
	int helperProcessStart;

/******************************************************************************/
/*                   MASTER PROCESS MAKES ALL CHOICES                         */
/******************************************************************************/
	if (rank == 0){
		for (a=0; a<n; a++){
			initial_cluster_assign[a] = 0;
		}

	  // all cluster centers must have moved less than 1/1000 for kmean to stop
	  while (largest_change > THRESHOLD){
			if (VERBOSE){
				currentStep++;
				printf("/************************************/\n");
				printf("/          CURRENT CYCLE: %d\t     /\n", currentStep);
				printf("/************************************/\n");
			}

			largest_change = 0.0;

	    // initialize initial_centers
	    for (a=0;a<k;a++){
	      for (b=0; b<dim;b++){
	        initial_centers[a*dim+b] = cluster_centers[a*dim+b];
	      }
	    }

			// send initial center data to other processes
			for (a=0; a<size; a++){
				MPI_Send(cluster_centers, NUMBER_OF_CENTERS*DIMENSIONS, MPI_DOUBLE, a, 4, MPI_COMM_WORLD);
			}

			do{
				// need to reset certain data
				for (a=0; a<k; a++){
					cluster_size[a] = 0;
					for (b=0; b<dim; b++){
						averages[a*dim+b] = 0.0;
					}
				}
				allCentersAssigned = 1;

				// assign each data point to the closest center
				printf("master calling assign points\n");
				assignPoints(data, cluster_centers, cluster_assign, rank * partialSize, partialSize, rank, size);
				printf("master called assign points\n");

				// figure out number of points assigned to each center
				for (a=0; a<n; a++){
					cluster_size[cluster_assign[a]]++;
				}

				for (a=0; a<k; a++){
					printf("%d\n", cluster_size[a]);
				}

				// check to make sure every center has at least 1 point assigned to it
				for (a=0; a<k; a++){
					if (cluster_size[a] == 0){
						// send message to other processes that they need to repeat
						recenterAgain[0] = 1;
						for (b=0; b<size; b++){
							MPI_Send(recenterAgain, 1, MPI_INT, b, 70, MPI_COMM_WORLD);
						}
						int farthestPoint;
						double greatestDistance = 0.0;
						allCentersAssigned = 0;
						// figure out which point is farthest away from all the centers
						for (b=0; b<n; b++){
							double tempDistance = distance(dim, &data[b*dim], &cluster_centers[a*dim]);
							if (tempDistance > greatestDistance){
								greatestDistance = tempDistance;
								farthestPoint = b;
							}
						}
						for (b=0; b<dim; b++){
							cluster_centers[a*dim+b] = data[farthestPoint*dim+b];
						}
					}
				}
			} while (allCentersAssigned == 0);

			// send message to other processes that they can proceed
			recenterAgain[0] = 0;
			for (a=0; a<size; a++){
				MPI_Send(recenterAgain, 1, MPI_INT, b, 70, MPI_COMM_WORLD);
			}

			for (a=0; a<size; a++){
				MPI_Send(cluster_centers, NUMBER_OF_CENTERS*DIMENSIONS, MPI_DOUBLE, a, 4, MPI_COMM_WORLD);
			}

			recenterCenters(data, cluster_centers, cluster_assign, 0, partialSize, 0, size);

	    // figure out which center moved the most and set it to largest_change
			for (a=0; a<k; a++){
				double distance_moved = 0.0;
				if (cluster_size[a] > 0){
					distance_moved = distance(dim, &cluster_centers[a*dim], &initial_centers[a*dim]);
					if (distance_moved > largest_change){
						largest_change = distance_moved;
					}
				}
			}
			printf("LARGEST CHANGE WAS %f\n", largest_change);
			if (largest_change > THRESHOLD){
				keepGoing[0] = 1;
				for (a=0; a<size; a++){
					MPI_Send(keepGoing, 1, MPI_INT, a, 99, MPI_COMM_WORLD);
				}
			}
	  }
		keepGoing[0] = 0;
		for (a=0; a<size; a++){
			MPI_Send(keepGoing, 1, MPI_INT, a, 99, MPI_COMM_WORLD);
		}
	}
/******************************************************************************/
/*                   OTHER PROCESSES ARE JUST HELPERS                         */
/******************************************************************************/
	else {
		/**************************************************************************/
		/*                  RECEIVE INITIAL CENTER DATA                           */
		/**************************************************************************/
		MPI_Recv(cluster_centers, NUMBER_OF_CENTERS*DIMENSIONS, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		while (keepGoing[0] == 1){
			/************************************************************************/
			/*              ASSIGN SUBSET OF POINTS TO CENTERS                      */
			/************************************************************************/
			do{
				printf("slave calling assign poins\n");
				assignPoints(data, cluster_centers, cluster_assign, rank * partialSize, partialSize, rank, size);
				// find out if master needed to reassign a center
				printf("slave called assign points\n");
				MPI_Recv(recenterAgain, 1, MPI_INT, 0, 70, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			} while(recenterAgain[0] == 1);
			/************************************************************************/
			/*                        RECEIVE FINAL CENTER DATA                     */
			/************************************************************************/
			MPI_Recv(cluster_centers, NUMBER_OF_CENTERS*DIMENSIONS, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			/************************************************************************/
			/*                        RECALCULATE CENTERS                           */
			/************************************************************************/
			recenterCenters(data, cluster_centers, cluster_assign, rank * partialSize, partialSize, rank, size);
			/************************************************************************/
			/*              DOES MASTER PROCESS WANT TO CONTINUE                    */
			/************************************************************************/
			MPI_Recv(keepGoing, 1, MPI_INT, 0, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}
}

double distance(int dim, double * datum1, double * datum2){
  int i;
  double dist = 0.0;

  for (i=0; i<dim; i++){
    dist += pow(datum1[i]-datum2[i], 2);
  }

  return sqrt(dist);
}


void recenterCenters(double * data, double * centers, short * centerAssign, int start, int size, int rank, int numProcesses){
	int a, b;
	int selectedCenter;
	int clusterSize[NUMBER_OF_CENTERS];
	double partialAverages[NUMBER_OF_CENTERS*DIMENSIONS];
	double totalAverages[NUMBER_OF_CENTERS*DIMENSIONS];
	for (a=0; a<NUMBER_OF_CENTERS*DIMENSIONS; a++){
		partialAverages[a] = 0;
	}
	for (a=0; a<NUMBER_OF_CENTERS; a++){
		clusterSize[a] = 0;
	}
	/****************************************************************************/
	/*              EACH PROCESS RECENTERS A SUBSET OF CENTERS                  */
	/****************************************************************************/
	for (a=start; a<start+size; a++){
		selectedCenter = centerAssign[a];
		clusterSize[selectedCenter]++;
		for (b=0; b<DIMENSIONS; b++){
			partialAverages[selectedCenter*DIMENSIONS+b] += data[a*DIMENSIONS+b];
		}
	}

	for (a=0; a<NUMBER_OF_CENTERS; a++){
		if (clusterSize[a] > 0){
			for (b=0; b<DIMENSIONS; b++){
				partialAverages[a*DIMENSIONS+b] = partialAverages[a*DIMENSIONS+b] / clusterSize[a];
			}
		}
	}
	/************************************************************************/
	/*          MASTER PROCESS NEEDS TO KNOW ALL NEW CENTERS                */
	/************************************************************************/
	if (rank == 0){
		for (a=0; a<NUMBER_OF_CENTERS*DIMENSIONS; a++){
			totalAverages[a] = partialAverages[a];
		}
		for (a=1; a<numProcesses; a++){
			MPI_Recv(partialAverages, NUMBER_OF_CENTERS*DIMENSIONS, MPI_DOUBLE, a, 50, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			for (b=0; b<NUMBER_OF_CENTERS*DIMENSIONS; b++){
				totalAverages[b] += partialAverages[b];
			}
		}
		for (a=0; a<NUMBER_OF_CENTERS*DIMENSIONS; a++){
			totalAverages[a] /= numProcesses;
		}
		for (a=0; a<NUMBER_OF_CENTERS*DIMENSIONS; a++){
			centers[a] = totalAverages[a];
		}
	} else {
		MPI_Send(partialAverages, NUMBER_OF_CENTERS*DIMENSIONS, MPI_DOUBLE, 0, 50, MPI_COMM_WORLD);
	}
}

void assignPoints(double * data, double * centers, short * centerAssign, int start, int size, int rank, int numProcesses){
	int a, b;
	short partialAssign[NUMBER_OF_POINTS];
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
		partialAssign[a] = currentClosestCenter;
	}

	if (rank == 0){
		// first assign it's own findings to complete assignment set
		for (a=0; a<size; a++){
			centerAssign[a] = partialAssign[a];
		}
		// then, receive data from other processes
		for (a=1; a<numProcesses; a++){
			MPI_Recv(partialAssign, size, MPI_SHORT, a, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			// NOTE a has changed
			for (b=0; b<size; b++){
				centerAssign[a*size+b] = partialAssign[b];
			}
		}
	}
	// All other processes
	else {
		// send results to master
		MPI_Send(partialAssign, size, MPI_SHORT, 0, 15, MPI_COMM_WORLD);
	}
	// impose a barrier so the pointers don't go out of scope until master is ready
	MPI_Barrier(MPI_COMM_WORLD);
}
