/* mpicc jacobi2.c -o jacobi2 */
/* or: cc jacobi2.c -o jacobi.c -lmpi */
/* mpirun -np m^2 jacobi2 m local_n */
/* Standard debugging run:
   mpirun -np 4 jacobi2 2 6 */

/*------------------------------------------------------------
   This is the parallel Jacobi iterations.
--------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define tol 0.0001
#define max_iter 10000

void Initialize(double**, int, int, int, int, int);
void rb(int local_n, double **x, int n, int row, int col, 
		     double **temp);
double Find_exact(int row, int col, int i, int j, int local_n, int m);
double error_exact(int local_n, double **x, int m, int row, int col);
double **alloc_2d_array(int m, int n);
void     free_2d_array(double **array);
int  is_bdry_point(int row, int col, int i, int j, int local_n, int m);
void print_values(int local_n, double **x, int m, int row, int col);

int main(int argc, char **argv){

  /* m = number of block rows = number of block columns */
  int i, row, col, myrank, m, n, dims[2], wrap_around[2], coords[2], local_n; 

  double start, timing, **x, **temp, error;

  MPI_Comm comm_2D;
  
  /* Start up MPI */
  MPI_Init(&argc, &argv);

  /* get command line arguments: m, local_n */
  if ( argc >= 2 )
    m = atoi(argv[1]);
  else
    m = 2;
  if ( argc >= 3 )
    local_n = atoi(argv[2]);
  else
    local_n = 5;

  printf("I am here!\n");

  /* Set up cartesian m x m grid */
  dims[0] = dims[1] = m;        
  wrap_around[0] = wrap_around[1] = 0; 
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, wrap_around, 0, &comm_2D);
  MPI_Comm_rank(comm_2D, &myrank);
  MPI_Cart_coords(comm_2D, myrank, 2, coords);
  row = coords[0];  col = coords[1];

    
  /* local_n = n/2 + 2; for m == 2 */
  n = m*(local_n-2);
  /* local_size = local_n * local_n; */

  x    = alloc_2d_array(local_n,local_n);
  temp = alloc_2d_array(local_n,local_n);
   
  /* Get the boundary conditions and initial guess. */                       
  Initialize(x, n, local_n, row, col, m);   
   
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  
  /* Perform parallel Jacobi iterations. */
  rb(local_n, x, m, row, col, temp);  
                     
  MPI_Barrier(MPI_COMM_WORLD); 
  timing = MPI_Wtime() - start;

  /* print_values(local_n, x, m, row, col); */

  if(myrank == 0)
    printf("timing for rb: %e\n",timing); 

  error = error_exact(local_n, x, m, row, col);
  if(myrank == 0)
    printf("total error after rb: %e\n",error);

  
  /* Clear communicator */ 
  MPI_Comm_free(&comm_2D);   
 
  /* Shut down MPI */
  MPI_Finalize();
} 

int rank_of(int row, int col, int m)
{
  return m*row + col;
}

#define square(z)   ((z)*(z))

void rb(int local_n, double **x_old,
		     int m, int row, int col, double **x_new)
{
  int i, j, n, tag1 = 1, tag2 = 2, num_iter = 0;
  double local_sum, sum, diff_norm, exact;
  
  MPI_Datatype col_mpi_t;
  MPI_Status status;  

  n = m*(local_n-2);

  /* Define col_mpi_t datatype. */
  MPI_Type_vector(local_n, 1, local_n, MPI_DOUBLE, &col_mpi_t);
  MPI_Type_commit(&col_mpi_t);
  
  do { /* Start iterations */
    num_iter ++;

    /* Set up the initial local_sum(Sum of 2_norm of (x_new-x_old)). */
    local_sum = 0;  
              
 /* For Red Points */                                  
    for(i = 1; i < local_n-1; i++){
      for(j = 1; j < local_n-1; j++){
         if((i + j) % 2 == 0) {
	/* Find new values. */
	x_new[i][j] = (x_old[i-1][j] + x_old[i+1][j] 
		       + x_old[i][j-1] + x_old[i][j+1]) * 0.25;

	/* Updates local_sum. */
	local_sum = local_sum + square(x_new[i][j]-x_old[i][j]); 
        }
      }
    }


    /* make sure that the local boundary has the old values */
    for ( i = 0; i < local_n; i++ ){
	x_new[i][0] = x_old[i][0];
	x_new[i][local_n-1] = x_old[i][local_n-1];
      }
    for ( j = 0; j < local_n; j++ ){
	x_new[0][j] = x_old[0][j];
	x_new[local_n-1][j] = x_old[local_n-1][j];
      }
 
    /* Send data to adjacent processors */
    if(row < m-1){ /* up */
      MPI_Send(&x_new[local_n-2][0], local_n, MPI_DOUBLE, 
	       rank_of(row+1,col,m), tag1, MPI_COMM_WORLD);
      MPI_Recv(&x_old[local_n-1][0], local_n, MPI_DOUBLE, 
	       rank_of(row+1,col,m), tag2, MPI_COMM_WORLD, &status);
    }
    if(row > 0){ /* down */
      MPI_Send(&x_new[1][0], local_n, MPI_DOUBLE, 
	       rank_of(row-1,col,m), tag2, MPI_COMM_WORLD);
      MPI_Recv(&x_old[0][0], local_n, MPI_DOUBLE, 
	       rank_of(row-1,col,m), tag1, MPI_COMM_WORLD, &status);
    }
    if(col < m-1){ /* right */
      MPI_Send(&x_new[0][local_n-2], 1, col_mpi_t, 
	       rank_of(row,col+1,m), tag1, MPI_COMM_WORLD);
      MPI_Recv(&x_old[0][local_n-1], 1, col_mpi_t,
	       rank_of(row,col+1,m), tag2, MPI_COMM_WORLD, &status);
    }
    if(col > 0){ /* left */
      MPI_Send(&x_new[0][1], 1, col_mpi_t, 
	       rank_of(row,col-1,m), tag2, MPI_COMM_WORLD);
      MPI_Recv(&x_old[0][0], 1, col_mpi_t, 
	       rank_of(row,col-1,m), tag1, MPI_COMM_WORLD, &status);
    }
    
/* Copy x_new into x_old and do iterations again. */
        for(i = 1; i < local_n-1; i++)
           for(j = 1; j < local_n-1; j++)
                x_old[i][j] = x_new[i][j];

/* For Black Points */

   for(i = 1; i < local_n-1; i++){
      for(j = 1; j < local_n-1; j++){
         if((i + j) % 2 != 0) {
	/* Find new values. */
	x_new[i][j] = (x_old[i-1][j] + x_old[i+1][j] 
		       + x_old[i][j-1] + x_old[i][j+1]) * 0.25;

	/* Updates local_sum. */
	local_sum = local_sum + square(x_new[i][j]-x_old[i][j]); 
        }
      }
    }

    /* make sure that the local boundary has the old values */
    for ( i = 0; i < local_n; i++ ){
	x_new[i][0] = x_old[i][0];
	x_new[i][local_n-1] = x_old[i][local_n-1];
      }
    for ( j = 0; j < local_n; j++ ){
	x_new[0][j] = x_old[0][j];
	x_new[local_n-1][j] = x_old[local_n-1][j];
      }
 
    /* Send data to adjacent processors */
    if(row < m-1){ /* up */
      MPI_Send(&x_new[local_n-2][0], local_n, MPI_DOUBLE, 
	       rank_of(row+1,col,m), tag1, MPI_COMM_WORLD);
      MPI_Recv(&x_old[local_n-1][0], local_n, MPI_DOUBLE, 
	       rank_of(row+1,col,m), tag2, MPI_COMM_WORLD, &status);
    }
    if(row > 0){ /* down */
      MPI_Send(&x_new[1][0], local_n, MPI_DOUBLE, 
	       rank_of(row-1,col,m), tag2, MPI_COMM_WORLD);
      MPI_Recv(&x_old[0][0], local_n, MPI_DOUBLE, 
	       rank_of(row-1,col,m), tag1, MPI_COMM_WORLD, &status);
    }
    if(col < m-1){ /* right */
      MPI_Send(&x_new[0][local_n-2], 1, col_mpi_t, 
	       rank_of(row,col+1,m), tag1, MPI_COMM_WORLD);
      MPI_Recv(&x_old[0][local_n-1], 1, col_mpi_t,
	       rank_of(row,col+1,m), tag2, MPI_COMM_WORLD, &status);
    }
    if(col > 0){ /* left */
      MPI_Send(&x_new[0][1], 1, col_mpi_t, 
	       rank_of(row,col-1,m), tag2, MPI_COMM_WORLD);
      MPI_Recv(&x_old[0][0], 1, col_mpi_t, 
	       rank_of(row,col-1,m), tag1, MPI_COMM_WORLD, &status);
    }


    /* Reduce local_sum in each proc to sum and broadcast sum. */
    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    /* diff_norm = ||x_new-x_old||_2/n */
    diff_norm = sqrt(sum)/n;
     
    /* Copy x_new into x_old and do iterations again. */
    for(i = 1; i < local_n-1; i++)
      for(j = 1; j < local_n-1; j++)
	x_old[i][j] = x_new[i][j];
    
  } while(num_iter < max_iter && tol < diff_norm); /* end of do-while */
 
 // Difference between true and approx rb
   for(i = 1; i < local_n-1; i++){
      for(j = 1;j < local_n-1; j++){
         exact = Find_exact(row, col, i, j, local_n, n);
         printf("e(%d,%d): %f \n",i,j, exact);
      }
   }



  if(rank_of(row,col,m) == 0 && tol > diff_norm )
    printf("num_iter: %d with diff_norm: %f\n", num_iter, diff_norm);
  else if(rank_of(row,col,m) == 0 && tol <= diff_norm)
    printf("Jacobi iteration does NOT converges\n");

  MPI_Type_free(&col_mpi_t);
}





/* error_exact -- computes error in comparison with the exact solution */
double error_exact(int local_n, double **x, int m, int row, int col)
{
  int i, j, n;
  double exact_sol, local_sum, sum;

  n = m*(local_n-2);
  local_sum = 0;
  for ( i = 0; i < local_n; i++ )
    for ( j = 0; j < local_n; j++ )
    {
      exact_sol = Find_exact(row, col, i, j, local_n, m);
      local_sum += square(x[i][j] - exact_sol);
    }
  MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(sum)/n;
}

void print_values(int local_n, double **x, int m, int row, int col)
{
  int i, j, i1,j1;

  for ( i = 0; i < m; i++ )
    for ( j = 0; j < m; j++ )
      {
	MPI_Barrier(MPI_COMM_WORLD);
	if ( i == row && j == col )
	  {
	    printf("\nLocal array for processor (%d,%d)\n",row,col);
	    for ( i1 = 0; i1 < local_n; i1++ )
	      {
		printf("Row %d:\n", i1);
		for ( j1 = 0; j1 < local_n; j1++ )
		  if ( (j1 % 5) == 4 )
		    printf("%12.7g\n", x[i1][j1]);
		  else
		    printf("%12.7g  ", x[i1][j1]);
		if ( (local_n % 5) != 0 )
		  printf("\n");
	      }
	  }
      }
}

/* This gives the exact solution */
double exact_func(double x, double y)
{
  return x*x - y*y;
}

/* These functions describe the geometry of the problem and its representation */

/* is_bdry_point -- returns TRUE (or non-zero) if (i,j) is a boundary point,
   and FALSE (zero) otherwise */
int is_bdry_point(int row, int col, int i, int j, int local_n, int m)
{
  if ( (row == 0 && i == 0) || (row == m-1 && i == local_n-1) )
    return 1;
  if ( (col == 0 && j == 0) || (col == m-1 && j == local_n-1) )
    return 1;
  return 0;
}

/* x_value -- returns the x co-ordinate of this point */
double x_value(int row, int i, int local_n, int m)
{
  int n;
  double h, x;

  n = m*(local_n-2);
  h = 1/(double)(n+1);
  return x = h*(row*(local_n-2)+i);
}

/* y_value -- returns the y co-ordinate of this point */
double y_value(int col, int j, int local_n, int m)
{
  int n;
  double h, y;

  n = m*(local_n-2);
  h = 1/(double)(n+1);
  return y = h*(col*(local_n-2)+j);
}

/* End of functions describing geometry */

void Initialize(double **u, int n, int local_n, int row, int col, int m)
{
  int i, j;
  double x, y;

  for(i = 0; i < local_n; i++){
    for ( j = 0; j < local_n; j++ ){
      if ( is_bdry_point(row,col,i,j,local_n,m) )
	{
	  x = x_value(row,i,local_n,m);
	  y = y_value(col,j,local_n,m);
	  u[i][j] = exact_func(x,y);
	}
      else
	u[i][j] = 0.0;
    }
  }
}


/* Find exact solution x^2 - y^2. */
double Find_exact(int row, int col, int i, int j, int local_n, int m)
{ 
  double x, y;

  x = x_value(row,i,local_n,m);
  y = y_value(col,j,local_n,m);
  return exact_func(x,y);
}  


/* alloc_2d_array -- allocate m x n array as a pointer to pointer to double */
double **alloc_2d_array(int m, int n)
{
  double **x;
  int i;

  x = (double **)malloc(m*sizeof(double *));
  x[0] = (double *)malloc(m*n*sizeof(double));
  for ( i = 1; i < m; i++ )
    x[i] = &x[0][i*n];
  return x;
}

/* free_2d_array -- de-allocate m x n array allocated by alloc_2d_array */
void free_2d_array(double **x)
{
  free(x[0]);
  free(x);
}
