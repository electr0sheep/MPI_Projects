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

#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>

#define DIMENSIONS 8
#define NUMBER_OF_POINTS 1000
#define NUMBER_OF_CENTERS 16
#define MAX_RANDOM_NUMBER 10000
#define THRESHOLD .001
#define VERBOSE 1
#define DEBUG 0
#define PRINT_BEGIN_AND_END_DATA 0
#define PRINT_CENTER_DATA 0
#define PRINT_CENTER_REASSIGN 1
#define PRINT_CLUSTER_SIZE 1
#define PRINT_ALL_POINTS_SAME_CENTER 1
#define PRINT_TIME_SPENT 1

void kmean(int, int, double *, short, short *, double *, int *);
double distance(int, double *, double *);

int main(int argc, char * argv[]) {
	int rank;
	int size;

  int i;
  int k;
  int increment = NUMBER_OF_POINTS / NUMBER_OF_CENTERS;
  double data[NUMBER_OF_POINTS*DIMENSIONS];
  short cluster_assign[NUMBER_OF_POINTS];
  double cluster_centers[NUMBER_OF_CENTERS*DIMENSIONS];
  int cluster_size[NUMBER_OF_CENTERS];

	time_t time1 = time(NULL);
	time_t time2;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  // fill data with random crap
  srand(50);
  for (i=0; i<NUMBER_OF_POINTS;i++){
    for (k=0;k<DIMENSIONS;k++){
      data[i*DIMENSIONS+k] = rand() % MAX_RANDOM_NUMBER;
    }
  }

  // assign each center to a data point
  for (i=0; i<NUMBER_OF_CENTERS; i++){
    for (k=0;k<DIMENSIONS; k++){
      cluster_centers[i*DIMENSIONS+k] = data[i*DIMENSIONS*increment+k];
    }
  }

	if (PRINT_BEGIN_AND_END_DATA){
		printf("/************************************/\n");
		printf("/      INITIAL CLUSTER DATA:\t     /\n");
		printf("/************************************/\n\n\n");
		for (i=0; i<NUMBER_OF_CENTERS;i++){
			printf("Cluster %d center:\n[\n", i);
			for (k=0; k<DIMENSIONS;k++){
				printf(" %f\n", cluster_centers[i+k]);
			}
			printf("]\nCluster %d size: %d\n", i, cluster_size[i]);
		}
	}

  kmean(DIMENSIONS, NUMBER_OF_POINTS, data, NUMBER_OF_CENTERS, cluster_assign, cluster_centers, cluster_size);

	if (PRINT_BEGIN_AND_END_DATA) {
		printf("/************************************/\n");
		printf("/        FINAL CLUSTER DATA:\t     /\n");
		printf("/************************************/\n\n\n");
		for (i=0; i<NUMBER_OF_CENTERS;i++){
			printf("Cluster %d center:\n[\n", i);
			for (k=0; k<DIMENSIONS;k++){
				printf(" %f\n", cluster_centers[i+k]);
			}
			printf("]\nCluster %d size: %d\n", i, cluster_size[i]);
		}
	}

	time2 = time(NULL);
	if (PRINT_TIME_SPENT){
		printf("COMPUTATION TOOK %f SECONDS\n", difftime(time2, time1));
	}

	MPI_Finalize();
	return 0;
}

void kmean(int dim, int n, double * data, short k, short * cluster_assign, double * cluster_centers, int * cluster_size){
  int a, b, c;
  double largest_change = DBL_MAX;
  double initial_centers[k*dim];
	double averages[k*dim];
	int currentStep = 0;
	int allCentersAssigned = 0;
	short initial_cluster_assign[n];
	int allPointsAssignedSameCenter = 0;

	for (a=0; a<n; a++){
		initial_cluster_assign[a] = 0;
	}

  // all cluster centers must have moved less than 1/1000 for kmean to stop
  while (largest_change > THRESHOLD){
		if (VERBOSE){
			currentStep++;
			printf("/************************************/\n");
			printf("/          CURRENT CYCLE: %d\t     /\n", currentStep);
			printf("/************************************/\n\n\n");
		}

		largest_change = 0.0;

    // initialize initial_centers
    for (a=0;a<k;a++){
      for (b=0; b<dim;b++){
        initial_centers[a*dim+b] = cluster_centers[a*dim+b];
      }
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
			// a is the element of data
	    for (a=0;a<n;a++){
	      // figure out which center is closest
				// b is the element of cluster_centers
				double closest_distance = DBL_MAX;
				short closest_center = -1;
	      for (b=0;b<k;b++){
					// c is the current dimension
	        for (c=0;c<dim;c++){
	          double dist = distance(dim, &data[a*dim], &cluster_centers[b*dim]);
	          if (dist < closest_distance){
	            closest_distance = dist;
	            closest_center = b;
	          }
	        }
	      }
				cluster_assign[a] = closest_center;
				cluster_size[closest_center]++;
	    }

			if (PRINT_CLUSTER_SIZE){
				for (a=0; a<k; a++){
					printf("CENTER %d\tHAS %d\tELEMENTS\n", a, cluster_size[a]);
				}
			}

			if (DEBUG){
				int sum = 0;
				for (a=0; a<k; a++){
					printf("CENTER %d POINTS: %d\n", a, cluster_size[a]);
					sum += cluster_size[a];
				}
				printf("SUM OF ALL CENTER POINTS: %d\n", sum);
			}

			// check to make sure every center has at least 1 point assigned to it
			for (a=0; a<k; a++){
				if (cluster_size[a] == 0){
					if (PRINT_CENTER_REASSIGN){
						printf("REASSIGNING CENTER\n");
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

    // center the centers to the data points assigned to it
		for (a=0; a<n; a++){
			int selectedCluster = cluster_assign[a];
			for (b=0; b<dim; b++){
				averages[selectedCluster*dim+b] += data[a*dim+b];
			}
		}

		for (a=0; a<k; a++){
			for (b=0; b<dim; b++){
				cluster_centers[a*dim+b] = averages[a*dim+b] / cluster_size[a];
			}
		}

		if (PRINT_CENTER_DATA){
			for (a=0; a<k; a++){
				printf("NEW CLUSTER CENTER FOR %d[\n", a);
				for (b=0; b<dim; b++){
					printf("%f\n", cluster_centers[a*dim+b]);
				}
				printf("]\n\n");
				printf("OLD CLUSTER CENTER FOR %d[\n", a);
				for (b=0; b<dim; b++){
					printf("%f\n", initial_centers[a*dim+b]);
				}
				printf("]\n\n");
				printf("DELTA FOR CENTER %d[\n", a);
				for (b=0; b<dim; b++){
					double delta = initial_centers[a*dim+b] - cluster_centers[a*dim+b];
					printf("%f\n", delta);
				}
				printf("]\n\n");
			}
		}

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
		if (PRINT_ALL_POINTS_SAME_CENTER){
			allPointsAssignedSameCenter = 1;
			for (a=0; a<n; a++){
				if (initial_cluster_assign[a] != cluster_assign[a]){
					allPointsAssignedSameCenter = 0;
				}
			}
			if (allPointsAssignedSameCenter){
				printf("ALL POINTS HAVE BEEN ASSIGNED TO THE SAME CENTER\n");
				printf("THEREFORE, THE CHANGE SHOULD BE 0\n");
			}
			for (a=0; a<n; a++){
				initial_cluster_assign[a] = cluster_assign[a];
			}
		}
		if (VERBOSE){
			printf("LARGEST CHANGE WAS %f\n\n", largest_change);
		}
  }
	if (VERBOSE){
		if (largest_change > 0.0){
			printf("THE THRESHOLD WAS REACHED BEFORE ALL POINTS WERE\n");
			printf("ASSIGNED TO THE SAME CENTER\n");
		}
	}
}

double distance(int dim, double * datum1, double * datum2){
  int i;
  double dist = 0.0;

  for (i=0; i<dim; i++){
    // add datum1[i]-datum2[i] raised to the power of 2 to dist
    dist += pow(datum1[i]-datum2[i], 2);
  }

  // distance formula = squareroot((x1-x2)^2 + (y1-y2)^2)
  // NOTE: in this case, we are dealing with unknown dimensions (up to 8)
  return sqrt(dist);
}
