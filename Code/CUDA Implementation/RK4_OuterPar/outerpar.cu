#include<stdio.h>
#define _USE_MATH_DEFINES
#include<math.h>
#include<float.h>
#include <cuda.h>
#include <device_functions.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <time.h>

typedef struct {
	double x;
	double y;
} point;

//CPU vars
//stepsizes
double h1; //Stepsize on level 1 (steps per year)
double h2; //Stepsize on level 2 (steps per year)
double h3; //Stepsize on level 3 (steps per year)
double g; //Years form the insurance holders death to when the payment starts
double r; //retirement age
double x; //Age at the time of calculation (t = 0)
double level2fullsteps; //since the fullsteps in the middle model is constant it can be calculated in the main() method

#define imin(a,b) (a<b?a:b)
int threadsPerBlock;
int blocksPerGrid;

bool shouldPrint = 0;

#define N 7
//x,r,g,h1,h2,h3,h2fullstepsizes
__constant__ double constData[N];

point Outer();
point OuterRK(double x, double y, double h, double *temp, int index);
double OuterDiff(double px, double py, double middley);
__global__ void Middle(double *outerx, double *temp);
__device__ point Inner(double eta, double t, double k);
__device__ point InnerRK(double h, point p, double eta, double t, double k, double x);
__device__ double InnerDiff(double t, double s, double fs, double k, double eta);
__device__ double MiddleDiff(double tau, double eat, double InnerY);
__device__ double r_d(double t);
__host__ double r_(double t);
__device__ double k(double tau);
__device__ double gTau(double tau);
__host__ double GmFemale(double t);
__device__ double GmMale(double t);
__device__ double Parabel(double t);
__device__ double f(double eta, double tau);
double roundWithString(double n);

int main(int argc, char *argv[])
{
	cudaSetDevice(0);
	clock_t start = clock();
	//CPU vars
	//stepsizes
	g = strtod(argv[1], NULL);
	r = strtod(argv[2], NULL);
	x = strtod(argv[3], NULL);
	h1 = strtod(argv[4], NULL);
	h2 = 2.0;
	h3 = strtod(argv[4], NULL);
	// h1 = 10.0; //Stepsize on level 1 (steps per year)
	// h2 = 2.0; //Stepsize on level 2 (steps per year)
	// h3 = 10.0; //Stepsize on level 3 (steps per year)
	// g = 10.0; //Years form the insurance holders death to when the payment starts
	// r = 65.0; //retirement age
	// x = 35.0; //Age at the time of calculation (t = 0)
	level2fullsteps = floor(119 * h2); //since the fullsteps in the middle model is constant it can be calculated in the main() method
	double h_data[N] = {x,r,g,h1,h2,h3,level2fullsteps};
	
	//Copy data to constData
	cudaMemcpyToSymbol(constData, h_data, N*sizeof(double));
	
	//Start solver
	point result = Outer();

	clock_t end = clock();
	double elapsedTime = (end-start)/(double)CLOCKS_PER_SEC;
	printf("%0.2f\n", elapsedTime);
	//printf("OuterPar\n");
	//printf("Execution time: %0.2f\n", elapsedTime);

	//print GPU result
	//printf("Result point: %.14f , %.14f \n" , result.x, result.y);
	//Prevent the cmd window from instantly closing
	//getc(stdin);

	//terminate
	return 0;
}

//the outer model
//Change outer.
//It needs to take a pointer to r, g and x and nextPoint, and change return type to void
point Outer(){
	
	double stepsize = -1 / h1; //since we are taking steps back the stepsize is negative
	
	int fullsteps = floor((120 - x) * h1); //the full amount of steps we need to take in this model
	
	//double firstStep = -(fullsteps * stepsize) - (120-x); //since the fullsteps is an int we need to take the remainder as a step first (if one exists)	
	int size = (fullsteps*2);

	//CPU vars:
	double *steps = new double[size];
	double *temp = new double[size];

	//GPU vars:
	double *d_steps;
	double *d_temp;

	//Prepare an array of all possible x values
	int stp = 0;
	int b; //fullstep counter
	for(b = 0; b <= size; b++)
	{
		//Take full step
		if(b%2 == 0){
			steps[b] = (120-x) + stp*(-1.0/h1);
		}
		//Take half step (value needed in OuterRK)
		else{
			steps[b] = ((120-x) + stp*(-1.0/h1))+((-1.0/h1)/2);
			stp += 1;
		}
	}

	//Set blocks pr grid and threads pr block
	blocksPerGrid = size;
	threadsPerBlock = 128;

	//printf("Blocks: %d\n", blocksPerGrid);
	//printf("Threads pr block: %d\n", threadsPerBlock);
	//printf("Ineffective threads in each block: %.1f\n", (level2fullsteps/threadsPerBlock)*threadsPerBlock-level2fullsteps);

	//Malloc and set memory on the GPU
	cudaMalloc((void**)&d_steps, size*sizeof(double));
	cudaMalloc((void**)&d_temp, size*sizeof(point));
	cudaMemset(d_steps, 0.0, size*sizeof(double));
	cudaMemset(d_temp, 0, size*sizeof(double));

	//Copy data to the GPU
	cudaMemcpy(d_steps, steps, size*sizeof(double), cudaMemcpyHostToDevice);

	//Start the kernel
	Middle<<<blocksPerGrid,threadsPerBlock, level2fullsteps*sizeof(double)>>>(d_steps, d_temp);
	//Check that everything went ok
	{
	    cudaError_t cudaerr = cudaDeviceSynchronize();
	    if (cudaerr != (cudaError_t)CUDA_SUCCESS)
	        printf("kernel launch failed with error \"%s\".\n",
	               cudaGetErrorString(cudaerr));
	}
	//copy intermediate results back to CPU
	cudaMemcpy(temp, d_temp, size*sizeof(double), cudaMemcpyDeviceToHost);

	point nextPoint = {120.0-x,0.0}; //set the startpoint

	int s; //fullstep counter
	//Run outer as normal using intermediate results from the GPU
	for(s = 0; s <size; s+=2)
	{
		nextPoint = OuterRK(steps[s], nextPoint.y, stepsize, temp, s);
	}
	cudaFree(d_temp);
	cudaFree(d_steps);
	return nextPoint;
}

point OuterRK(double x, double y, double h, double *temp, int index){
 double k1 = h * OuterDiff(x, y, temp[index]);

 double k2 = h * OuterDiff(x + h/2.0, y + k1/2, temp[index+1]);

 double k3 = h * OuterDiff(x + h/2.0, y + k2/2, temp[index+1]);
 
 double k4 = h * OuterDiff(x + h, y + k3, temp[index+2]);
 
 double newy = y + k1/6 + k2/3 + k3/3 + k4/6;

 point resultPoint = {x + h,newy};
 return resultPoint;
}

//The differential equation for the outer model
double OuterDiff(double px, double py, double middley)
{
	return r_(px) * py - GmFemale(x + px) * (middley - py);
}

//the middle model
__global__ void Middle(double *outerx, double *temp){
	
	extern __shared__ double kSum[];

	int tid = threadIdx.x;
	double stepsize = -1.0 / constData[4]; //since we are taking steps back the stepsize is negative
	//since the fullsteps in the middle model is constant it can be calculated in the main() method
	const int fullsteps = constData[6];
	double gt = outerx[blockIdx.x];
	while(tid<fullsteps){
		double tau = constData[0] + gt;
		double kk = k(tau);
		double eta = (120.0+(tid*stepsize));
		double k1 = stepsize * MiddleDiff(tau, eta, Inner(eta, gt, kk).y);
		double k2 = stepsize * MiddleDiff(tau, eta + stepsize/2, Inner(eta + stepsize/2, gt, kk).y);		
		double k4 = stepsize * MiddleDiff(tau, eta + stepsize, Inner(eta + stepsize, gt, kk).y);
		kSum[tid] = k1/6+k2/3+k2/3+k4/6;
		//kSum[tid] = 0.0;
		tid = tid + blockDim.x;
	}
	tid = threadIdx.x;
	__syncthreads();

	if(tid == 0){
		double y = 0.0;
		int i;
		for(i = fullsteps-1; i>=0; i--){
			y = y+kSum[i];
		}
		temp[blockIdx.x] = y;
	}
}

//The differential equation for the middle model
__device__ double MiddleDiff(double tau, double eta, double innerY)
{
	return -1 * gTau(tau) * f(eta,tau) * innerY;
}

__device__ point Inner(double eta, double t, double k){
	
	double stepsize = -1.0 / constData[5]; //since we are taking steps back the stepsize is negative
	point nextPoint = {120-eta, 0}; //set the startpoint
	int fullsteps = floor((120 - eta) * constData[5]); //the full amount of steps we need to take in this model
	
	double firstStep = -(fullsteps * stepsize) - (120-eta); //since the fullsteps is an int we need to take the remainder as a step first (if one exists)
	nextPoint = InnerRK(firstStep, nextPoint, eta, t, k, (120-eta));

	//Solve the differential equation with the Runge-Kutta solver
	int s; //fullstep counter
	for(s = 0; s < fullsteps; s++)
	{	
		nextPoint = InnerRK(stepsize, nextPoint, eta, t, k, (((120.0-eta)+firstStep)+s*stepsize));
	}
	return nextPoint;
}

//the runge-kutta solver for the inner model
__device__ point InnerRK(double h, point p, double eta, double t, double k, double x){
	
	double k1 = h * InnerDiff(t, x, p.y, k, eta);
	
	double k2 = h * InnerDiff(t, (x + h/2), (p.y + k1/2), k, eta);		
	
	double k3 = h * InnerDiff(t, (x + h/2), (p.y + k2/2), k, eta);
	
	double k4 = h * InnerDiff(t, (x + h), (p.y + k3), k, eta);	
	
	double y = p.y + k1/6 + k2/3 + k3/3 + k4/6;
	
	point resultPoint = {x + h,y};
	return resultPoint;
}

//The differential equation for the inner model
__device__ double InnerDiff(double t, double s, double fs, double k, double eta)
{
	return r_d(t + s) * fs - (s >= k ? 1 : 0) - GmMale(eta + s) * (0 - fs);
}

//FUNCTIONS
//helper function to gTau. Taken from collectiveHelp.txt
__device__ double Parabel(double t){
	return fmax((15.0 - t) * (t - 120.0), 0.0);
}

//Function to determine g(tau)
__device__ double gTau(double tau)
{
	//taken from collectiveHelp.txt
	double marriageProbabilityPeak = 0.9;
	double scalefactor = Parabel(67.5) / marriageProbabilityPeak;
	
	return Parabel(tau) / scalefactor;
}

//Function to determine f(eta,tau)
__device__ double f(double eta, double tau){
	//taken from collectiveHelp.txt
	double sigma = 3.0;
    double preFactorInNormalDensity = 1.0 / (sqrt(2.0 * M_PI) * sigma);
    double factorInExponent = 1.0 / (2.0 * sigma * sigma);

	return 
      (eta <= 0.0 || eta >= 120.0) 
      ? 0.0
      : preFactorInNormalDensity * exp(-1.0 * (tau - eta) * (tau - eta) * factorInExponent);
}

//Function to determine k
__device__ double k(double tau) {
	double r = constData[1];
	double g = constData[2];
	if(tau < r) return g;
	if(r <= tau && tau < r + g) return (r + g - tau);
	if(r + g <= tau) return 0.0;
	return 0.0;
}

//R function - interest rate
__device__ double r_d(double t) {
	return 0.05;
}

//R function - interest rate
__host__ double r_(double t) {
	return 0.05;
}

// Gompertz-Makeham mortality intensities for Danish women
__host__ double GmFemale(double t) {
    return 0.0005 + pow(10.0, 5.728 - 10 + 0.038*(t));
}

// Gompertz-Makeham mortality intensities for Danish men
__device__ double GmMale(double t) {
    return 0.0005 + pow(10.0, 5.880 - 10 + 0.038*(t));
}

double roundWithString(double n)
{
char buffer [50];
sprintf (buffer, "%.13f", n);
return atof(buffer);
}
