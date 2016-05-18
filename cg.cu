#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "common.h"

/*
   Globale Variablen stehen in allen Funktionen zur Verfuegung.
   Achtung: Das gilt *nicht* fuer Kernel-Funktionen!
*/
int Nx, Ny, npts;
int *active;

/*
   Fuer die Koordinaten:
      i = 0,1,...,Nx+1
      j = 0,1,...,Ny+1
   wird der fortlaufenden Index berechnet
*/
int coord2index(int i, int j)
{
   return j*(Nx+2) + i;
}

/*
   Das Flag-Array der aktiven/inneren Punkte wird gesetzt.
*/
void active_pts()
{
   int idx,i,j;

   active=(int*)malloc(npts*sizeof(int));

   idx=0; // fortlaufender Index
   for (j=0; j<Ny+2; j++)
   {
      for (i=0; i<Nx+2; i++)
      {
         if ((i==0)||(j==0)||(i==Nx+1)||(j==Ny+1))
            active[idx]=0; // Randpunkt
         else
            active[idx]=1; // innerer Punkt

         idx+=1;
      }
   }
}

/*
   Der Vektor p wird im Inneren auf zufaellige Werte gesetzt
*/
void random_vector(double *p)
{
   int idx;

   for(idx = 0; idx < npts; idx++)
   {
      if (active[idx])
         p[idx] = (double)(rand() & 0xFF ) / 10.0;
   }
}

/*
   Das Flag-Array der aktiven/inneren Punkte wird als
   2D Gitter ausgegeben.
*/
void print_active()
{
   int i,j,idx;

   printf("active points:\n");
   idx=0;
   for (j=0; j<Ny+2; j++)
   {
      printf("  ");
      for (i=0; i<Nx+2; i++)
      {
         printf("%d ",active[idx]);
         idx+=1;
      }
      printf("\n");
   }
}


/*
   Norm-Quadrat vom Vektor v.
*/
double norm_sqr(double *v)
{
   int idx;
   double r=0.0;
   for (idx=0; idx<npts; idx++)
   {
      r+=v[idx]*v[idx];
   }
   return r;
}

/*
   Der Vektor p wird als 2D Gitter fuer i,j<=16 ausgegeben. Es werden innere/aktive
   und, falls flag>0, auch die aeusseren Punkte ausgegeben.
*/
void print_vector(const char *name, double *p, int flag)
{
   int i,j,idx;
   double nrm;

   printf("%s = \n",name);
   idx=0;
   for (j=0; j<Ny+2; j++)
   {
      if (j>16)
      {
         printf("  ...\n");
         break;
      }
      printf("  ");
      for (i=0; i<Nx+2; i++)
      {
         if ((i<16)&&((flag>0)||(active[idx])))
            printf("%.2f ",p[idx]);
         if (i==16)
            printf("...");
         idx+=1;
      }
      printf("\n");
   }
   nrm=norm_sqr(p);
   printf("||%s|| = %.8f\n",name,sqrt(nrm));
}

void laplace_2d(double *w, double *v)
{
	int i,j;
	for (i=1; i<Nx+1; i++)
	{
		for (j=1; j<Ny+1; j++)
			w[coord2index(i,j)] = 4*v[coord2index(i,j)] - (v[coord2index(i-1,j)]+v[coord2index(i+1,j)]+v[coord2index(i,j-1)]+v[coord2index(i,j+1)]);
	}
}

__global__ void laplace_2d_gpu(double *w, double *v, const int nx, const int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	if (ix>0 && ix<(nx+1) && iy>0 && iy<(ny+1))
	{
		unsigned int idx = iy*(blockDim.x * gridDim.x) + ix;
		w[idx] = 4*v[idx] - (v[idx-1] + v[idx+1] + v[(idx-(gridDim.x*blockDim.x))] + v[(idx+(gridDim.x*blockDim.x))]);
	}
}

double vec_scalar(double *w, double *v)
{
	double scalar = .0;
	int i;
	for (i=0; i<npts; i++)
		scalar += w[i]*v[i];
	return scalar;
}

void vec_add(double *sum, double *w, double a, double *v)
{
	int i;
	for (i=0; i<npts; i++)
		sum[i] = w[i] + a*v[i];
}

__global__ void vec_add_gpu(double *sum, double *w, double a, double *v, const int nx, const int ny)
{	
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	if (ix>0 && ix<(nx+1) && iy>0 && iy<(ny+1))
	{
		unsigned int idx = iy*(blockDim.x * gridDim.x) + ix;
		sum[idx] = w[idx] + a*v[idx];
	}
}

int main(int argc, char **argv)
{
   printf("%s Starting...\n", argv[0]);

   // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
    
   int nBytes, k, kmax;
   double *s, *v, *x, *r, *v_backup;
   double tol;
   double rnorm, rnorm_alt, alpha, beta;

   // Globale Variablen setzen:
   // Anzahl der Inneren Punkte in x- und y-Richtung
   Nx=8;
   Ny=8;
   if (argc>1)
   {
     sscanf(argv[1],"%d",&Nx);
     if (Nx % 32 != 0)
     {
       printf("Die Eingabe (Nx+2) muss ein Vielfaches von 32 sein!\n");
       return (1);
     }
     Nx -= 2;
     Ny = Nx;
   }
   
   // Gesamtanzahl der Gitterpunkte
   npts=(Nx+2)*(Ny+2);
   // Aktive Punkte - Array
   active_pts();

   // Speicherbedarf pro Vektor in Byte
   nBytes=npts*sizeof(double);

   // Toleranz & Iterationsgrenze festlegen
   tol = 1e-6;
   kmax = 1e3;
   k = 0;
   
   // Speicher für Vektoren allozieren
   s=(double*)malloc(npts*sizeof(double));
   x=(double*)malloc(npts*sizeof(double));
   v=(double*)malloc(npts*sizeof(double));
   r=(double*)malloc(npts*sizeof(double));
   v_backup=(double*)malloc(npts*sizeof(double));

   // auf Null setzen
   memset(s, 0, nBytes);
   memset(x, 0, nBytes);
   memset(v, 0, nBytes);
   memset(r, 0, nBytes);
   
   // Aktive Punkte ausgeben
   if ((Nx<=16)&&(Ny<=16))
      print_active();

   // Einheitsvektor
   v[coord2index(Nx/2+1,Nx/2+1)]=1.0; // v=0, ausser am Gitterpunkt (Nx/2+1,Ny/2+1)
   print_vector("v",v,1);

   // Zufaelliger Vektor
   random_vector(v);
   print_vector("v",v,1);
   memcpy(v_backup,v,nBytes);
   
  // Nullter Iterationsschritt
	double iStart = seconds();
	laplace_2d(s,v);
	const double time_laplace_host = seconds() - iStart; // Zeitmessung fuer 3.2
	print_vector("s_host",s,1); // Ausgabe fuer Aufgabe 3.1.1
	
	rnorm_alt = norm_sqr(v);
	alpha = rnorm_alt/vec_scalar(s,v);
	const double alpha_backup = alpha;
	vec_add(x,x,alpha,v);
	iStart = seconds();
	vec_add(r,v,(-alpha),s);
	const double time_vec_add_host = seconds() - iStart; // Zeitmessung fuer 3.2
	rnorm = norm_sqr(r);

//test	print_vector("x",x,1);

   // Iteration
	while (k<kmax && rnorm>tol)
	{
//test		printf("Norm: %.4f \n",rnorm);
		beta = rnorm/rnorm_alt;
		vec_add(v,r, beta,v);

		rnorm_alt = rnorm;
		laplace_2d(s,v);
		alpha = vec_scalar(v,r)/vec_scalar(v,s);
		vec_add(x,x,alpha,v);
		vec_add(r,r,(-alpha),s);
		rnorm = norm_sqr(r);

		k++;
	}
   // Ausgabe Ergebnis
   printf("Anzahl Iterationen: %d \n",k);
   print_vector("x_Ergebnis",x,1);

   // auf Null setzen
   memset(s, 0, nBytes);
   memset(x, 0, nBytes);
   memset(r, 0, nBytes);
   
   // Speicher auf GPU allozieren
   double *s_gpu, *x_gpu, *v_gpu, *r_gpu;
   CHECK(cudaMalloc((void**)&s_gpu, nBytes));
   CHECK(cudaMalloc((void**)&x_gpu, nBytes));
   CHECK(cudaMalloc((void**)&v_gpu, nBytes));
   CHECK(cudaMalloc((void**)&r_gpu, nBytes));
   
   // GPU-Blocks vorbereiten
   int bdim = Nx+2;
   if ((Nx+2)>32 && (Ny+2)>32)
     bdim = 32;
   
     dim3 block(bdim,bdim);
     dim3 grid(((Nx+1+block.x)/block.x), ((Ny+1+block.y)/block.y));

   printf("\n GPU-Berechnung\n Grid-Dim: %d x %d , Block-Dim: %d x %d \n", grid.x, grid.y,block.x,block.y);
   
   // GPU-Rechnungen fuer Aufgabe 3.2
   CHECK(cudaMemcpy(s_gpu, s, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(v_gpu, v_backup, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(r_gpu, r, nBytes, cudaMemcpyHostToDevice));

   iStart = seconds();
   laplace_2d_gpu<<<grid,block>>>(s_gpu,v_gpu,Nx,Ny);
   CHECK(cudaDeviceSynchronize());
   const double time_laplace_gpu = seconds() - iStart; // Zeitmessung fuer 3.2
   
   iStart = seconds();
   vec_add_gpu<<<grid,block>>>(r_gpu,v_gpu,(-alpha_backup),s_gpu,Nx,Ny);
   CHECK(cudaDeviceSynchronize());
   const double time_vec_add_gpu = seconds() - iStart; // Zeitmessung fuer 3.2
   
   // check kernel error
   CHECK(cudaGetLastError());
   
   CHECK(cudaMemcpy(s, s_gpu, nBytes, cudaMemcpyDeviceToHost));
   print_vector("s_gpu",s,1);
   
   printf("\n Speedup Laplace: %.6f \n",(time_laplace_host/time_laplace_gpu));
   printf(" Speedup Vec_add: %.6f \n",(time_vec_add_host/time_vec_add_gpu));

   // free device memory
   CHECK(cudaFree(s_gpu));
   CHECK(cudaFree(x_gpu));
   CHECK(cudaFree(v_gpu));	
   CHECK(cudaFree(r_gpu));	

   // free host memory
   free(active);
   free(s);
   free(x);
   free(v);
   free(r);
   free(v_backup);

   // reset device
   CHECK(cudaDeviceReset());

   return (0);
}
