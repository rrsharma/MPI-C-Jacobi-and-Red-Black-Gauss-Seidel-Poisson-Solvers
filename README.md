# MPI-C-Jacobi-and-Red-Black-Gauss-Seidel-Poisson-Solvers
MPI / C : Jacobi and Red-Black Gauss Seidel Poisson Solvers
mpicc jacobi2.c -o jacobi2
or: cc jacobi2.c -o jacobi.c -lmpi
mpirun -np m^2 jacobi2 m local_n
Standard debugging run:
mpirun -np 4 jacobi2 2 6
