/* Hello World program */

#include<stdio.h>

//CONSTANTS
static double tau;		//Dødsfaldstidspunkt
static double r;		//Pensionsalder
static double g;		//År fra forsikredes død til udbetaling startes
static double x;		//Alder på beregninstidspunktet (t = 0)
static double t;		//Beregningstidspunktet
static double n;		//Ægtefælles alder
static double h;		//Stepsize

typedef struct {
	double x;
	double y;
} point;

double f1(double s, double fs);
double r_(double t);
double my(double s);
double k(double tau, double r, double g);
point RK(point p1, double (*f1)(double s, double fs));

int main()
{
	tau = 40.0;
	r = 60.0;
	g = 10.0;
	x = 30.0;
	t = 0.0;
	n = 32.0;
	h = 1.0;

	point p1 = { 10.0, 1.0 };
	point p2 = RK(p1, &f1);
	printf("p2.x = %f; p2.y = %f", p2.x, p2.y);
	
	return 0;

}

//Inderste model - Runge-Kutta
point RK(point p1, double (*f)(double s, double fs) ) {

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
}

//Inderste model
double f1(double s, double fs) {
	return r_(t + s) * fs - (s >= k(tau, r, g) ? 1 : 0) - my(s) * (0 - fs);
}

//FUNCTIONS

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
	return 20;
}