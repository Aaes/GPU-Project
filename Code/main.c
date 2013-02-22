#include<stdio.h>

//CONSTANTS
static double tau;		//Dødsfaldstidspunkt
static double r;		//Pensionsalder
static double g;		//År fra forsikredes død til udbetaling startes
static double x;		//Alder på beregninstidspunktet (t = 0)
static double t;		//Beregningstidspunktet
static double n;		//Ægtefælles alder
static double h;		//Stepsize
static double eta = 0; 	//ægtefældes alder

static int counter1 = 0; //test counter, for testing purposes
static int counter2 = 0; //test counter, for testing purposes
static int counter3 = 0; //test counter, for testing purposes

typedef struct {
	double x;
	double y;
} point;

double f1(double s, double fs);
double f2(double n, double fn);
double f3(double t, double ft);
double r_(double t);
double my(double s);
double k(double tau, double r, double g);
point RK(point p1, double (*f1)(double s, double fs));
double gTau(double tau);
point level3 (point startPoint);
point level2 (point startPoint);
point level1 (point startPoint);

int main()
{
	tau = 50.0;
	r = 60.0;
	g = 10.0;
	x = 30.0;
	t = 0.0;
	n = 32.0;
	h = -0.5;

	point p1 = { 1.0, 1.0 };
	point p2 = level1(p1);
	printf("p2.x = %f; p2.y = %f", p2.x, p2.y);
	//point p3 = level2(p2);
	//printf("\np3.x = %f; p3.y = %f", p3.x, p3.y);
	printf("\nStepsize: %f", h);
	printf("\nAntal level 1 beregninger: %d", counter1);
	printf("\nAntal level 2 beregninger: %d", counter2);
	printf("\nAntal level 3 beregninger: %d", counter3);
	return 0;

}

//Runge-Kutta
point RK(point p1, double (*f)(double x, double y) ) {

	double k1 = h * f(p1.x, p1.y);
	
	double k2 = h * f(p1.x + h/2, p1.y + k1/2);
	
	double k3 = h * f(p1.x + h/2, p1.y + k2/2);
	
	double k4 = h * f(p1.x + h, p1.y + k3);
	
	double y = p1.y + k1/6 + k2/3 + k3/3 + k4/6;
	
	point p = {p1.x + h,y};
	return p;
	
	/* IKKE RIGTIG KODE
	//Forskrift for tangenten (step 1)
	double slope1 = f(p1.x, p1.y);				//Hældning i startpunktet
	double b0 = p1.y - slope1 * p1.x;			//Finder B i Y = AX + B
	double y1 = slope1 * (p1.x + h/2) + b0;		//Finder Y-værdien på tangenten i X = (X0 + h/2)

	//Forskrift for tangenten (step 2)
	double slope2 = f(p1.x + h/2, y1);			//Hældning i punkt nr. 2
	double b1 = y1 - slope2 * (p1.x + h/2);		//Finder B i Y = AX + B
	double y2 = slope2 * (p1.x + h/2) + b1;		//Finder Y-værdien på ny tangent i X = (X0 + h/2)

	//Forskrift for tangenten (step 3)
	double slope3 = f(p1.x + h/2, y2);			//Hældning i punkt nr. 3
	double b2 = y2 - slope3 * (p1.x + h/2);		//Finder B i Y = AX + B
	double y3 = slope3 * (p1.x + h/2) + b2;		//Finder Y-værdien på ny tangent i X = (X0 + h/2)

	//Hældning i sidste (4.) punkt(step 4)
	double slope4 = f1(p1.x + h, y3);			//Hældning i punkt nr. 4

	//Weighted average af de 4 hældninger
	double avg = slope1/6 + slope2/3 + slope3/3 + slope4/6;

	//Endelig forskrift til at finde det næste punkt på grafen
	double b = p1.y - avg * p1.x;				//Finder B i Y = AX + B
	double y = avg * (p1.x + h) + b;			//Finder Y-værdien på det endelige punkt (X = X0 + h)

	point p = { p1.x + h, y };
	return p;
	*/
}

point level3 (point startPoint)
{	
	point nextPoint = startPoint;

	double s;
	for(s = 120 - eta; s >= 0; s = s + h)
	{
		eta = s;
		nextPoint = RK(nextPoint, f1);
		counter3++;
	}
	
	return nextPoint;
}

point level2 (point startPoint)
{
	point nextPoint = startPoint;
	
	double n;
	for(n = 120; n > 0; n = n + h)
	{
		nextPoint = RK(nextPoint, f2);
		counter2++;
	}
	
	return nextPoint;
} 

point level1 (point startPoint)
{
	point nextPoint = startPoint;
	
	double t;
	for(t = 120-x; t > 0; t = t + h)
	{
		nextPoint = RK(nextPoint, f3);
		counter1++;
	}
	
	return nextPoint;
}

//Inderste model - funktion
double f1(double s, double fs) {
	return r_(t + s) * fs - (s >= k(tau, r, g) ? 1 : 0) - my(s) * (0 - fs);
}

//Mellemste model - funktion
double f2(double n, double fn)
{
	point p = {n,fn};
	
	return -1 * gTau(tau) * fn * level3(p).y; 
}

//Yderste model - funktion
double f3(double t, double ft)
{
	point p = {t, ft};
	
	return r_(t) * ft - my(x + t) * (level2(p).y - ft);
}

//FUNCTIONS

//Function to determine g(tau)
double gTau(double tau)
{
	return 1;
}

//Function to determine k
double k(double tau, double r, double g) {
	if(tau < r) return g;
	if(r <= tau && tau < r + g) return (r + g - tau);
	if(r + g <= tau) return 0.0;
	return 0.0;
}

//R function (formodentlig pensionsalder)
double r_(double t) {
	return 0.05;
}

//Dødeligheden for en n + s årig t+s år efter
//den dato dødelighederne tager udgangspunkt i
double my(double s) {
	return 0.5;
}