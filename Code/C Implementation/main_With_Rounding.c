#include<stdio.h>
#include<math.h>
#include<float.h>
#include<stdlib.h>

//CONSTANTS
static double r;		//retirement age
static double g;		//Years form the insurance holders death to when the payment starts
static double x;		//Age at the time of calculation (t = 0)
static double h1;		//Stepsize on level 1 (steps per year)
static double h2;		//Stepsize on level 2 (steps per year)
static double h3;		//Stepsize on level 3 (steps per year)

static double level2fullsteps; //since the fullsteps in the middle model is constant it can be calculated in the main() method

static int counter1 = 0; //test counter, for testing purposes
static int counter2 = 0; //test counter, for testing purposes
static int counter3 = 0; //test counter, for testing purposes

static int printOuter = 0; 	//Print flag for the outer level - testing purposes
static int printMiddle = 0;	//Print flag for the middle level - testing purposes
static int printInner = 0;	//Print flag for the inner level - testing purposes

typedef struct {
	double x;
	double y;
} point;

point* Outer();
point OuterRK(double h, point p1);
point Middle(double t);
point MiddleRK(double h, point p, double eta, double t);
point Inner(double eta, double t, double k);
point InnerRK(double h, point p, double eta, double t, double k);
double InnerDiff(double t, double s, double fs, double k, double eta);
double MiddleDiff(double tau, double eat, double InnerY);
double OuterDiff(double px, double py);
double r_(double t);
double k(double tau, double r, double g);
double gTau(double tau);
double GmFemale(double t);
double GmMale(double t);
double Parabel(double t);
double f(double eta, double tau);

double roundWithString(double value);

int main()
{
	//stepsizes
	h1 = 10.0;
	h2 = 2.0;
	h3 = 10.0;
	
	//constants
	g = 31.0;
	r = 79.0;
	x = 100.0;
	
	level2fullsteps = floor(119 * h2); //since the fullsteps in the middle model is constant it can be calculated in the main() method
	
	point* results = Outer();
	
	//result prints
	printf("Result point: %.14f , %.14f \n" , results[0].x, results[0].y);
	printf("Outer stepsize = %.1f, antal beregninger = %d\n", h1, counter1);
	printf("Middle stepsize = %.1f, antal beregninger = %d\n", h2, counter2);
	printf("Inner stepsize = %.1f, antal beregninger = %d\n", h3, counter3);
	
	return results[0].y; //returns the y-value for the last point. Used for comparison.
}

//the outer model
point* Outer(){
	
	double stepsize = -1 / h1; //since we are taking steps back the stepsize is negative
	point nextPoint = {120.0-x,0.0}; //set the startpoint

	int fullsteps = floor((120 - x) * h1); //the full amount of steps we need to take in this model
	point* resultArray = malloc(fullsteps+1 * sizeof(point)); //initialize the resultArray. We added the +1 to make space for the initialstep

	double firstStep = -(fullsteps * stepsize) - (120-x); //since the fullsteps is an int we need to take the remainder as a step first (if one exists)
		
	nextPoint = OuterRK(firstStep, nextPoint);
	resultArray[fullsteps] = nextPoint;
		
	if(printOuter) printf("Point initial Step (%.10f, %.10f)\n",nextPoint.x, nextPoint.y);

	//Solve the differential equation with the Runge-Kutta solver
	int s; //fullstep counter
	for(s = 0; s < fullsteps; s++)
	{
		if(printOuter) printf("Point %d (%.10f, %.10f)\n",counter1,nextPoint.x, nextPoint.y);
		nextPoint = OuterRK(stepsize, nextPoint);
		resultArray[fullsteps-1-s] = nextPoint;
		counter1++;
	}
	
	return resultArray;
}

//the runge-kutta solver for the outer model
point OuterRK(double h, point p1){
	
	point p = p1;
	p.x = roundWithString(p.x);
	
	if(printOuter) printf("	k1 = %f * ",h);
	double k1 = h * OuterDiff(p.x, p.y);
	if(printOuter) printf(" = %f\n",k1);
	
	if(printOuter) printf("	k2 = %f * ",h);
	double k2 = h * OuterDiff(p.x + h/2, p.y + k1/2);	
	if(printOuter) printf(" = %f\n",k2);	
	
	if(printOuter) printf("	k3 = %f * ",h);
	double k3 = h * OuterDiff(p.x + h/2, p.y + k2/2);		
	if(printOuter) printf(" = %f\n",k3);
	
	if(printOuter) printf("	k4 = %f * ",h);
	double k4 = h * OuterDiff(p.x + h, p.y + k3);
	if(printOuter) printf(" = %f\n",k4);
	
	double y = p.y + k1/6 + k2/3 + k3/3 + k4/6;
	
	point resultPoint = {p.x + h,y};
	return resultPoint;
}

//The differential equation for the outer model
double OuterDiff(double px, double py)
{
	if(printOuter) printf("%f * %f - %f * (%f - %f)",r_(px) , py , GmFemale(x + px) , Middle(px).y , py);
	return r_(px) * py - GmFemale(x + px) * (Middle(px).y - py);
}

//the middle model
point Middle(double t){
	
	double stepsize = -1 / h2; //since we are taking steps back the stepsize is negative
	point nextPoint = {120.0,0.0}; //set the startpoint
	
	int fullsteps = level2fullsteps; //since the fullsteps in the middle model is constant it can be calculated in the main() method
	
	//Solve the differential equation with the Runge-Kutta solver
	int s; //fullstep counter
	double n = 120; // eta
	for(s = 0; s < fullsteps; s++)
	{
		if(printMiddle) printf("Middle Point %d (%.10f,%.10f)\n",counter2, nextPoint.x, nextPoint.y);
		nextPoint = MiddleRK(stepsize, nextPoint, n, t);
		n += stepsize;
		counter2++;
	}
	//printf("(Middle call g=%f, r=%f, tau=%.12f, t=%.12f, result=%.14f)\n", g,r,x+t,t, nextPoint.y);
	return nextPoint;
}

//the runge-kutta solver for the middle model
point MiddleRK(double h, point p, double eta, double t){
	
	double tau = x + t;
	
	double kk = k(tau,r,g);

	if(printMiddle) printf("	k1 = %f *",h);
	double k1 = h * MiddleDiff(tau, eta, Inner(eta, t, kk).y);		
	if(printMiddle) printf(" = %.14f\n",k1);
	
	if(printMiddle) printf("	k2 = %f *",h);
	double k2 = h * MiddleDiff(tau, eta + h/2, Inner(eta + h/2, t, kk).y);		
	if(printMiddle) printf(" = %.14f\n",k2);
	
	if(printMiddle) printf(" 	k3 = %f *",h);
	double k3 = k2; // since the equation for k3 is exactly the same as k2 we set them equal to each other
	//double k3 = h * MiddleDiff(tau, eta + h/2, Inner(eta + h/2, t, kk).y); 
	if(printMiddle) printf(" = %.14f\n",k3);
	
	if(printMiddle) printf("	k4 = %f *",h);
	double k4 = h * MiddleDiff(tau, eta + h, Inner(eta + h, t, kk).y);
	if(printMiddle) printf(" = %.14f\n",k4);
	
	double y = p.y + k1/6 + k2/3 + k3/3 + k4/6;
	
	point resultPoint = {p.x + h,y};
	return resultPoint;
}

//The differential equation for the middle model
double MiddleDiff(double tau, double eta, double innerY)
{
	if(printMiddle) printf("-1 * %f * %.10f * %.14f",-gTau(tau) , f(eta,tau) , innerY);
	return -1 * gTau(tau) * f(eta,tau) * innerY;
}

//the inner model
point Inner(double eta, double t, double k){
	
	double stepsize = -1 / h3; //since we are taking steps back the stepsize is negative
	point nextPoint = {120-eta, 0}; //set the startpoint
	
	int fullsteps = floor((120 - eta) * h3); //the full amount of steps we need to take in this model

	double firstStep = -(fullsteps * stepsize) - (120-eta); //since the fullsteps is an int we need to take the remainder as a step first (if one exists)
		
	nextPoint = InnerRK(firstStep, nextPoint, eta, t, k);

	//Solve the differential equation with the Runge-Kutta solver
	int s; //fullstep counter
	for(s = 0; s < fullsteps; s++)
	{
		if(printInner) printf("Inner Point %d (%.10f,%.10f)\n",counter3, nextPoint.x, nextPoint.y);
		nextPoint = InnerRK(stepsize, nextPoint, eta, t, k);
		counter3++;
	}

	return nextPoint;
}

//the runge-kutta solver for the inner model
point InnerRK(double h, point p, double eta, double t, double k){
	
	if(printInner) printf("		k1 = %f *",h);
	double k1 = h * InnerDiff(t, p.x, p.y, k, eta);
	if(printInner) printf(" = %.14f\n",k1);
	
	if(printInner) printf("		k2 = %f *",h);
	double k2 = h * InnerDiff(t, (p.x + h/2), (p.y + k1/2), k, eta);
	if(printInner) printf(" = %.14f\n",k2);		
	
	if(printInner) printf("		k3 = %f *",h);
	double k3 = h * InnerDiff(t, (p.x + h/2), (p.y + k2/2), k, eta);
	if(printInner) printf(" = %.14f\n",k3);
	
	if(printInner) printf("		k4 = %f *",h);
	double k4 = h * InnerDiff(t, (p.x + h), (p.y + k3), k, eta);
	if(printInner) printf(" = %.14f\n",k4);	
	
	double y = p.y + k1/6 + k2/3 + k3/3 + k4/6;
	
	point resultPoint = {p.x + h,y};
	return resultPoint;
}

//The differential equation for the inner model
double InnerDiff(double t, double s, double fs, double k, double eta)
{
	if(printInner) printf("%f * %f - %d - %f * (0 - %f)",r_(t + s) ,fs , (s >= k ? 1 : 0) , GmMale(eta + s) , fs);
	return r_(t + s) * fs - (s >= k ? 1 : 0) - GmMale(eta + s) * (0 - fs);
}

double roundWithString(double n)
{
	char buffer [50];
	sprintf (buffer, "%.13f", n);
   return atof(buffer);
}

//FUNCTIONS
//helper function to gTau. Taken from collectiveHelp.txt
double Parabel(double t){
	return fmax((15.0 - t) * (t - 120.0), 0.0);
}

//Function to determine g(tau)
double gTau(double tau)
{
	//taken from collectiveHelp.txt
	double marriageProbabilityPeak = 0.9;
	double scalefactor = Parabel(67.5) / marriageProbabilityPeak;
	
	return Parabel(tau) / scalefactor;
}

//Function to determine f(eta,tau)
double f(double eta, double tau){
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
double k(double tau, double r, double g) {
	if(tau < r) return g;
	if(r <= tau && tau < r + g) return (r + g - tau);
	if(r + g <= tau) return 0.0;
	return 0.0;
}

//R function - interest rate
double r_(double t) {
	return 0.05;
}

// Gompertz-Makeham mortality intensities for Danish women
double GmFemale(double t) {
    return 0.0005 + pow(10, 5.728 - 10 + 0.038*(t));
}

// Gompertz-Makeham mortality intensities for Danish men
double GmMale(double t) {
    return 0.0005 + pow(10, 5.880 - 10 + 0.038*(t));
}
