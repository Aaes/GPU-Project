#include<stdio.h>
#define _USE_MATH_DEFINES
#include<math.h>
#include<float.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>

typedef struct {
	double x;
	double y;
} point;

#define N 7
//x,r,g,h1,h2,h3,h2fullstepsizes
__constant__ double constData[N];

__global__ void Outer(point *result);
__device__ point OuterRK(double h, point p);
__device__ point Middle(double t);
__device__ point MiddleRK(double h, point p, double eta, double t);
__device__ point Inner(double eta, double t, double k);
__device__ point InnerRK(double h, point p, double eta, double t, double k);
__device__ double InnerDiff(double t, double s, double fs, double k, double eta);
__device__ double MiddleDiff(double tau, double eat, double InnerY);
__device__ double OuterDiff(double px, double py);
__device__ double r_(double t);
__device__ double k(double tau);
__device__ double gTau(double tau);
__device__ double GmFemale(double t);
__device__ double GmMale(double t);
__device__ double Parabel(double t);
__device__ double f(double eta, double tau);

int main()
{
	//CPU vars
	//stepsizes
	double h1 = 1.0; //Stepsize on level 1 (steps per year)
	double h2 = 2.0; //Stepsize on level 2 (steps per year)
	double h3 = 1.0; //Stepsize on level 3 (steps per year)
	double g = 31.0; //Years form the insurance holders death to when the payment starts
	double r = 79.0; //retirement age
	double x = 110.0; //Age at the time of calculation (t = 0)
	double level2fullsteps = floor(120 * h2) - h2; //since the fullsteps in the middle model is constant it can be calculated in the main() method
	double h_data[N] = {x,r,g,h1,h2,h3,level2fullsteps};
	point p = {0.0, 0.0};
	point *result = &p;

	//GPU vars
	point *d_result;
	
	//allocate __device__ memory for variables
	cudaMalloc( (void**)&d_result, sizeof(point));
	
	//Copy data to constData
	cudaMemcpyToSymbol(constData, h_data, N*sizeof(double));
	
	//Start kernel
	Outer<<<1,1>>>(d_result);
	
	//Check that everything went ok
	{
	    cudaError_t cudaerr = cudaDeviceSynchronize();
	    if (cudaerr != CUDA_SUCCESS)
	        printf("kernel launch failed with error \"%s\".\n",
	               cudaGetErrorString(cudaerr));
	}
	//Copy the result of running the kernel back to the CPU
	cudaMemcpy(result, d_result, sizeof(point), cudaMemcpyDeviceToHost);
	
	p = *result;
	//print GPU result
	printf("Result point: %.14f , %.14f \n" , p.x, p.y);
	//Prevent the cmd window from instantly closing
	getc(stdin);

	//terminate
	return 0;
}

//the outer model
//Change outer.
//It needs to take a pointer to r, g and x and nextPoint, and change return type to void
__global__ void Outer(point *result){
	
	double stepsize = -1 / constData[3]; //since we are taking steps back the stepsize is negative
	point nextPoint = {120.0-constData[0],0.0}; //set the startpoint
	
	int fullsteps = floor((120 - constData[0]) * constData[3]); //the full amount of steps we need to take in this model
	
	double firstStep = -(fullsteps * stepsize) - (120-constData[0]); //since the fullsteps is an int we need to take the remainder as a step first (if one exists)	
	nextPoint = OuterRK(firstStep, nextPoint);

	//Solve the differential equation with the Runge-Kutta solver
	int s; //fullstep counter
	for(s = 0; s < fullsteps; s++)
	{
		nextPoint = OuterRK(stepsize, nextPoint);
	}
	
	//printf("Nextpoint: %.14f , %.14f \n" , nextPoint.x, nextPoint.y);
	
	*result = nextPoint;
}

//the runge-kutta solver for the outer model
__device__ point OuterRK(double h, point p){
	
	double k1 = h * OuterDiff(p.x, p.y);
	
	double k2 = h * OuterDiff(p.x + h/2, p.y + k1/2);		
	
	double k3 = h * OuterDiff(p.x + h/2, p.y + k2/2);		

	double k4 = h * OuterDiff(p.x + h, p.y + k3);

	double y = p.y + k1/6 + k2/3 + k3/3 + k4/6;
	
	point resultPoint = {p.x + h,y};
	return resultPoint;
}

//The differential equation for the outer model
__device__ double OuterDiff(double px, double py)
{
	return r_(px) * py - GmFemale(constData[0] + px) * (Middle(px).y - py);
}

//the middle model
__device__ point Middle(double t){
	
	double stepsize = -1 / constData[4]; //since we are taking steps back the stepsize is negative
	point nextPoint = {120.0,0.0}; //set the startpoint
	
	int fullsteps = constData[6]; //since the fullsteps in the middle model is constant it can be calculated in the main() method
	
	//Solve the differential equation with the Runge-Kutta solver
	int s; //fullstep counter
	double n = 120; // eta
	for(s = 0; s < fullsteps; s++)
	{
		nextPoint = MiddleRK(stepsize, nextPoint, n, t);
		n += stepsize;
	}
	return nextPoint;
}

//the runge-kutta solver for the middle model
__device__ point MiddleRK(double h, point p, double eta, double t){
	
	double tau = constData[0] + t;
	
	double kk = k(tau);

	double k1 = h * MiddleDiff(tau, eta, Inner(eta, t, kk).y);		
	
	double k2 = h * MiddleDiff(tau, eta + h/2, Inner(eta + h/2, t, kk).y);		
	
	double k3 = h * MiddleDiff(tau, eta + h/2, Inner(eta + h/2, t, kk).y);
	
	double k4 = h * MiddleDiff(tau, eta + h, Inner(eta + h, t, kk).y);
	
	double y = p.y + k1/6 + k2/3 + k3/3 + k4/6;
	
	point resultPoint = {p.x + h,y};
	return resultPoint;
}

//The differential equation for the middle model
__device__ double MiddleDiff(double tau, double eta, double innerY)
{
	return -1 * gTau(tau) * f(eta,tau) * innerY;
}

//the inner model
__device__ point Inner(double eta, double t, double k){
	
	double stepsize = -1 / constData[5]; //since we are taking steps back the stepsize is negative
	point nextPoint = {120-eta, 0}; //set the startpoint
	
	int fullsteps = floor((120 - eta) * constData[5]); //the full amount of steps we need to take in this model

	double firstStep = -(fullsteps * stepsize) - (120-eta); //since the fullsteps is an int we need to take the remainder as a step first (if one exists)
		
	nextPoint = InnerRK(firstStep, nextPoint, eta, t, k);

	//Solve the differential equation with the Runge-Kutta solver
	int s; //fullstep counter
	for(s = 0; s < fullsteps; s++)
	{
		nextPoint = InnerRK(stepsize, nextPoint, eta, t, k);
	}

	return nextPoint;
}

//the runge-kutta solver for the inner model
__device__ point InnerRK(double h, point p, double eta, double t, double k){
	
	double k1 = h * InnerDiff(t, p.x, p.y, k, eta);
	
	double k2 = h * InnerDiff(t, (p.x + h/2), (p.y + k1/2), k, eta);	
	
	double k3 = h * InnerDiff(t, (p.x + h/2), (p.y + k2/2), k, eta);
	
	double k4 = h * InnerDiff(t, (p.x + h), (p.y + k3), k, eta);
	
	double y = p.y + k1/6 + k2/3 + k3/3 + k4/6;
	
	point resultPoint = {p.x + h,y};
	return resultPoint;
}

//The differential equation for the inner model
__device__ double InnerDiff(double t, double s, double fs, double k, double eta)
{
	return r_(t + s) * fs - (s >= k ? 1 : 0) - GmMale(eta + s) * (0 - fs);
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
__device__ double r_(double t) {
	return 0.05;
}

// Gompertz-Makeham mortality intensities for Danish women
__device__ double GmFemale(double t) {
    return 0.0005 + pow(10, 5.728 - 10 + 0.038*(t));
}

// Gompertz-Makeham mortality intensities for Danish men
__device__ double GmMale(double t) {
    return 0.0005 + pow(10, 5.880 - 10 + 0.038*(t));
}
