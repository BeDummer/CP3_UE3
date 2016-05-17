#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
void print_vector(char *name, double *p, int flag)
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

int main(int argc, char **argv)
{
   printf("%s Starting...\n", argv[0]);

   int nBytes, k, kmax;
   double *s, *v, *x, *r;
   double tol;
   double rnorm, rnorm_alt, alpha, beta;

   // Globale Variablen setzen:
   // Anzahl der Inneren Punkte in x- und y-Richtung
   Nx=8;
   Ny=8;
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
   // auf Null setzen
   memset(s, 0, nBytes);
   memset(x, 0, nBytes);
   memset(v, 0, nBytes);

   // Aktive Punkte ausgeben
   if ((Nx<=16)&&(Ny<=16))
      print_active();

   // Einheitsvektor
   v[coord2index(Nx/2+1,Nx/2+1)]=1.0; // v=0, ausser am Gitterpunkt (Nx/2+1,Ny/2+1)
   print_vector("v",v,1);

   // Zufaelliger Vektor
   random_vector(v);
   print_vector("v",v,1);

  // Nullter Iterationsschritt
	print_vector("s",s,1);

	laplace_2d(s,v);
	print_vector("s",s,1);
	
	rnorm_alt = norm_sqr(v);
	alpha = rnorm_alt/vec_scalar(s,v);
	vec_add(x,x,alpha,v);
	vec_add(r,v,(-alpha),s);
	rnorm = norm_sqr(v);

	print_vector("x",x,1);

   // Iteration
	while (k<kmax && rnorm>tol)
	{
		printf("Norm: %.4f \n",rnorm);
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
   // Ausgabe x
	printf("k = %d \n",k);
	print_vector("x",x,1);

   free(active);
   free(s);
   free(x);
   free(v);
   free(r);

   return (0);
}
