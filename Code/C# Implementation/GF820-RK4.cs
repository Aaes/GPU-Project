// Simple C# implementation of Runge-Kutta 4 solver, used to implement
// GF820

// Part of the Actulus project.

// sestoft@itu.dk * 2013-02-14, 2013-03-01

// DO NOT DISTRIBUTE: Contains project-internal information.

using System;
using System.Diagnostics;

// ============================================================

// General n-state fixed-step RK4 solver, works for any number n of
// equations.  The derivatives are given by function dV of type
// Action<double, double[], double[]>.  Slightly encapsulated (OO)
// version that reuses intermediate arrays.  This solver is used for
// all three levels of integration in GF820.

public class Rk4 {
  // Setup of the integrator; the arrays must have length n
  private readonly Action<double, double[],double[]> dV;
  private readonly int n;

  // These are used only in method OneStep; allocated here for reuse.
  private readonly double[] k1, k2, k3, k4, tmp;

  public Rk4(Action<double, double[],double[]> dV, int n) {
    this.dV = dV;
    if (n < 1)
      throw new Exception("There must be at least one equation");
    this.n = n;
    k1 = new double[n];
    k2 = new double[n];
    k3 = new double[n];
    k4 = new double[n];
    tmp = new double[n];
  }

  // Solve, and return the final (year b) function estimates (reserves).

  public double[] SolveFinal(double a, double b, int steps, double[] Va) {
    if (Va.Length != n)
      throw new Exception(String.Format("Va Length {0} not n {1}", Va.Length, n));
    double[] result = new double[n];
    double h = -1.0 / steps;
    Array.Copy(Va, result, Va.Length);  // result := Va
    int fullsteps = (int)(Math.Floor((a - b) * steps));
    double t = b - fullsteps * h; // May be different from a
    // Perhaps take an initial step from a to t, length a-t
    if (a - t > 1E-10) 
      OneStep(t - a, a, result);
    // Regular h-length steps
    for (int s=0; s<fullsteps; s++) { 
      OneStep(h, t, result);
      t += h;      
    }
    return result;
  }

  // Solve, and return all annual reserves

  public double[][] SolveAll(int a, int b, int steps, double[] Va) {
    if (Va.Length != n)
      throw new Exception(String.Format("Va Length {0} not n {1}", Va.Length, n));
    double[][] result = new double[a-b+1][];
    for (int y=a; y>=b; y--) 
      result[y-b] = new double[n];
    double h = -1.0 / steps;
    result[a-b] = Va;
    double[] v = new double[n];
    Array.Copy(Va, v, v.Length); // v := Va
    for (int y=a; y>b; y--) { 
      double t = y;
      for (int s=0; s<steps; s++) { 	// Integrate backwards over [y,y-1]
	OneStep(h, t, v);
	t += h;      
      }
      Array.Copy(v, result[y-1-b], v.Length); // result[y-1-b] := v
    }
    return result;
  }

  // Integrate from t to t+h 
  private void OneStep(double h, double t, double[] v) {
    dV(t, v, k1);
    sax(h, k1, k1);
    saxpy(0.5, k1, v, tmp);
    dV(t + h/2, tmp, k2);
    sax(h, k2, k2);
    saxpy(0.5, k2, v, tmp);
    dV(t + h/2, tmp, k3);
    sax(h, k3, k3);
    saxpy(1, k3, v, tmp);
    dV(t + h, tmp, k4);
    sax(h, k4, k4);
    saxpy(1/6.0, k4, v, tmp);
    saxpy(2/6.0, k3, tmp, tmp);
    saxpy(2/6.0, k2, tmp, tmp);
    saxpy(1/6.0, k1, tmp, v);
  }

  // sax = scalar a times x array, imperative version
  static void sax(double a, double[] x, double[] res) {
    if (x.Length != res.Length)
      throw new Exception("sax: lengths of x and res differ");
    for (int i=0; i<x.Length; i++)
      res[i] = a * x[i];
  }

  // saxpy = scalar a times x array plus y array, imperative version
  static void saxpy(double a, double[] x, double[] y, double[] res) {
    if (x.Length != y.Length)
      throw new Exception("saxpy: lengths of x and y differ");
    if (x.Length != res.Length)
      throw new Exception("saxpy: lengths of x and res differ");
    for (int i=0; i<x.Length; i++)
      res[i] = a * x[i] + y[i];
  }
}

// ============================================================

// Implementation of GF820 

class CalculationSpecifications {
  static readonly int steps = 2; // Per year

  static readonly double interestrate = 0.05;

  static double indicator(bool b) {
    return b ? 1.0 : 0.0;
  }

  // Gompertz-Makeham mortality intensities for Danish women
  // From Actulus/Server/TestUtilities/GompertzMakehamExtensions.cs
  static double GmFemale(double t) {
    return 0.0005 + Math.Pow(10, 5.728 - 10 + 0.038*(t));
  }

  // Gompertz-Makeham mortality intensities for Danish men
  static double GmMale(double t) {
    return 0.0005 + Math.Pow(10, 5.880 - 10 + 0.038*(t));
  }

  static double rate(double t) { 
    return interestrate;    // Fixed interest rate
    // return rFsa(t);      // Finanstilsynet's rate curve
  }

  // The Danish FSA yield curve (Finanstilsynets rentekurve).
  // Data from 2011-11-16 
  static readonly double[] 
    ts = new double[] { 
      0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 },
    rs = new double[] { 
      1.146677033, 1.146677033, 1.146677033, 1.340669678, 1.571952911, 1.803236144, 
      2.034519377, 2.26580261, 2.497085843, 2.584085843, 2.710085843, 2.805085843, 
      2.871485843, 2.937885843, 3.004285843, 3.070685843, 3.137085843, 3.136485843, 
      3.135885843, 3.135285843, 3.134685843, 3.134085843, 3.113185843, 3.092285843, 
      3.071385843, 3.050485843, 3.029585843, 3.008685843, 2.987785843, 2.966885843, 
      2.945985843, 2.925085843
    };

  // Get discount rate at time t by linear interpolation into (ts,rs); then
  // compute the instantaneous forward rate as described in 
  // https://wiki.actulus.dk/Documentation-CalculationPlatform-YieldCurves.ashx

  // This method uses binary search, which is needlessly slow because
  // the t values are monotonically decreasing.  Hence it would be
  // faster to keep the most recent index m into the ts/rs arrays and
  // decrement m only when t < ts[m].  It would also be easier to get
  // wrong, because it relies on many assumptions, so for now we stick
  // to the binary search.
 
  static double rFsa(double t) { 
    // Requires ts non-empty and elements strictly increasing.
    int last = ts.Length-1;
    if (t <= ts[0])
      return Math.Log(1 + rs[0]/100);
    else if (t >= ts[last])
      return Math.Log(1 + rs[last]/100);
    else {
      int a = 0, b = last;
      // Now a < b (bcs. ts must have more than 1 element) and ts[a] < t < ts[b]
      while (a+1 < b) {
	// Now a < b and ts[a] <= t < ts[b]
	int i = (a+b)/2;
	if (ts[i] <= t)
	  a = i;
	else // t < ts[i]
	  b = i;
      }
      // Now a+1>=b and ts[a] <= t < ts[b]; so a!=b and hence a+1 == b <= last
      int m = a;
      double tm = ts[m], tm1 = ts[m+1];
      double rm = rs[m] / 100, rm1 = rs[m+1] / 100;
      double Rt = (rm * (tm1 - t) + rm1 * (t - tm)) / (tm1 - tm);
      return Math.Log(1 + Rt) + t / (tm1 - tm) * (rm1 - rm) / (1 + Rt);
    }
  }

  static void Print(double[][] result) {
    for (int y=0; y<result.Length; y++) {
      Console.Write("{0,3}:", y);
      for (int i=0; i<result[y].Length; i++)
	Console.Write("  {0,20:F16}", result[y][i]);
      Console.WriteLine();
    }
  }

  // GF820 computations.  Notation as in Jeppe Woetmann Nielsen's note
  // on 820 dated 2013-01-25.  Female 35 year old insured life (method
  // Outer), unknown male spouse (method Inner).

  static double Inner(double eta, double k, double t) {
    Rk4 solver = new Rk4(
      (double s, double[] V, double[] res) =>
      { res[0] = rate(t+s) * V[0] - indicator(s >= k) - GmMale(eta+s) * (0 - V[0]); }, 
      1);
    double[] reserves = solver.SolveFinal(120-eta, 0, steps, new double[] { 0.0 });
    // Reserve in state 0:
    return reserves[0];
  }

  private static double Parabel(double t) {
    return Math.Max((15.0 - t) * (t - 120.0), 0.0);
  }

  // From collectiveHelp.txt
  private static readonly double marriageProbabilityPeak = 0.9;  
  private static readonly double scaleFactor = Parabel(67.5) / marriageProbabilityPeak;

  // Adapted from CollectiveSpouseParameters.cs, renamed from g to gp.
  static double gp(double tau) {
    return Parabel(tau) / scaleFactor;
  }
 
  // Adapted from CollectiveSpouseParameters.cs
  static double h(double eta, double tau) {
    double sigma = 3.0;  // From collectiveHelp.txt
    var preFactorInNormalDensity = 1.0 / (Math.Sqrt(2 * Math.PI) * sigma);
    var factorInExponent = 1.0 / (2.0 * sigma * sigma);
    return 
      (eta <= 0.0 || eta >= 120.0)
      ? 0.0
      : preFactorInNormalDensity * Math.Exp(-1.0 * (tau - eta) * (tau - eta) * factorInExponent);
  }

  static double Middle(int g, int r, double tau, double t) {
    double k = (tau < r) ? g : (tau < r+g) ? r+g-tau : 0;
    Rk4 solver = new Rk4(
      (double eta, double[] V, double[] res) =>
      { res[0] = -gp(tau) * h(eta, tau) * Inner(eta, k, t); },
      1);
    double[] reserves = solver.SolveFinal(120, 1, /* steps = */ 2, new double[] { 0.0 });
    // Integral in state 0:
    return reserves[0];
  }

  static double[][] Outer(int g, int r, int x) {
    Rk4 solver = new Rk4(
      (double t, double[] V, double[] res) =>
      { 
    	Console.WriteLine(t);
    	res[0] = rate(t) * V[0] - GmFemale(x+t) * (Middle(g,r,x+t,t) - V[0]); 
      },
      1);
    double[][] reserves = solver.SolveAll(120-x, 0, steps, new double[] { 0.0 });
    return reserves;
  }

  public static void Main(String[] args) {
    // From collectiveHelp.txt
    int g = 10, r = 65, x = 35 /* age */;
    double[][] reserves = Outer(g, r, x); 
    // Hardcoded value calculated by Actulus
    //Print(new double[][] { new double[] { 0.61831213598263246 } });
	Console.WriteLine(reserves[0][0]);
    //Print(reserves);
  }
}
