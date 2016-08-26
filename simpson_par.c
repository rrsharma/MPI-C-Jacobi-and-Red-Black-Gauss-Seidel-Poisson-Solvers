/***************************************************************/
/* Rahil Sharma                                                */
/* HPPC Exam Part-1: Sequential (a)                            */
/* Date 29th October, 2014                                     */
/* Due date: 3rd November, 2014                                */
/*                                                             */
/***************************************************************/

#include <stdio.h>
#include <math.h>
#include <mpi.h> 

float Simpson(float  a, float  b, int n, float h) {
   float integral;
   float x;
   float buffer;      /* Temporary calculation storage */
   int i;
   float f(float x);  /* Function we shall integrate */

   integral = f(a) + f(b); /* initializing integral to f(x_0) + f(x_n) */
   buffer = 0.0;           /* initializing buffer */  
   
   for (i = 1; i <= n - 1; i += 2) { /*note: n should always be even*/
      x = a + (i * h);
      buffer += f(x);
   }
   integral += 4 * buffer; /* all odd positions have a multiple 4 */
   buffer = 0.0;   /* reinitialize buffer */

   for (i = 2; i <= n - 2; i += 2) {
      x = a + (i * h);
      buffer += f(x); 
   }
   integral += 2 * buffer; /* all even positions have a multiple 2 */
   
   integral *= h;
   integral /= 3;

   return integral;
}   /* end of Simpson */


 /*******************************************************************/
 /************* USING "Get_data" GIVEN IN THE CLASS *****************/
 /*******************************************************************/
 /* Function Get_data                                               */
 /* Reads in the user input a, b, and n.                            */
 /* Input parameters:                                               */
 /*     1.  int my_rank:  rank of current process.                  */
 /*     2.  int p:  number of processes.                            */
 /* Output parameters:                                              */
 /*     1.  float* a_ptr:  pointer to left endpoint a.              */
 /*     2.  float* b_ptr:  pointer to right endpoint b.             */
 /*     3.  int* n_ptr:  pointer to number of trapezoids.           */
 /* Algorithm:                                                      */
 /*     1.  Process 0 prompts user for input and                    */
 /*         reads in the values.                                    */
 /*     2.  Process 0 sends input values to other                   */
 /*         processes.                                              */
 /*******************************************************************/
void Get_data(float* a_ptr, float* b_ptr, int* n_ptr, int my_rank, int p, int argc, char* argv[]) {

    int source = 0;    /* All local variables used by */
    int dest;          /* MPI_Send and MPI_Recv       */
    int tag;
    MPI_Status status;

    if (my_rank == 0) {
        /* printf("Enter a, b, and n\n"); */
        /* scanf("%f %f %d", a_ptr, b_ptr, n_ptr); */
	sscanf(argv[1], "%f", a_ptr);
	sscanf(argv[2], "%f", b_ptr);
	sscanf(argv[3], "%d", n_ptr);
    }
    /*********** Distributing a, b and n using BROADCAST ****************/
    MPI_Bcast(a_ptr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_ptr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
} /* end of Get_data */

float f(float x) {   /* Function to evaluate (Hardwired --> [e^(-x^2)]) */
    float return_val;
    return_val = expf(-(x*x));
    return return_val;
} /* enf of f */

main(int argc, char** argv) {
    int         my_rank;   /* My process rank           */
    int         p;         /* The number of processes   */
    float       a;         /* Left endpoint             */
    float       b;         /* Right endpoint            */
    int         n;         /* Number of trapezoids      */
    float       h;         /* Trapezoid base length     */
    float       local_a;   /* Left endpoint my process  */
    float       local_b;   /* Right endpoint my process */
    int         local_n;   /* Number of trapezoids for  */
                           /* my calculation            */
    float       integral;  /* Integral over my interval */
    float       total;     /* Total integral            */
    int         source;    /* Process sending integral  */
    int         dest = 0;  /* All messages go to 0      */
    int         tag = 0;
    double      elapsed_time;
    MPI_Status  status;

    void Get_data(float* a_ptr, float* b_ptr, int* n_ptr, int my_rank, int p, int argc, char *argv[]);
    float Simpson(float  a, float  b, int n, float h);  /* Calculate local integral  */

    /* Let the system do what it needs to start up MPI */
    MPI_Init(&argc, &argv);


    /* Get my process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out how many processes are being used */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    Get_data(&a, &b, &n, my_rank, p, argc, argv);

    h = (b-a)/n;    /* h is the same for all processes */
    local_n = n/p;  /* So is the number of subintervals */

    /* Length of each process' interval of 
     * integration = local_n*h.  So my interval
     * starts at: */
    local_a = a + my_rank*local_n*h;
    local_b = local_a + local_n*h;
    
    /* Calculating time : start*/
        MPI_Barrier(MPI_COMM_WORLD);
        elapsed_time = - MPI_Wtime();

    integral = Simpson(local_a, local_b, local_n, h);
    
    /* Calculating time : end*/
        elapsed_time +=  MPI_Wtime();

    /*********** Reduction operation ************/
     /* Adding up integrals calculated by each process */
    MPI_Reduce(&integral, &total, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   
    if (my_rank == 0) {
      printf("With n = %d subintervals, our estimate\n", n);
      printf("of the integral from %f to %f = %f\n", a, b, total); 
      printf("Time taken %f = ", elapsed_time);
    }

    /* Shut down MPI */
    MPI_Finalize();
} /*  main  */

