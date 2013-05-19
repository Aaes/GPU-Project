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
      t = b + (s - fullsteps) * h;
      OneStep(h, t, result);
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
  static int steps = 2; // Per year

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
    	// Console.WriteLine(t);
    	res[0] = rate(t) * V[0] - GmFemale(x+t) * (Middle(g,r,x+t,t) - V[0]); 
      },
      1);
    double[][] reserves = solver.SolveAll(120-x, 0, steps, new double[] { 0.0 });
    return reserves;
  }

  public static void Main(String[] args) {
    // From collectiveHelp.txt
	Stopwatch stopWatch = new Stopwatch();
	stopWatch.Start();
    int g = Convert.ToInt32(args[0]), r = Convert.ToInt32(args[1]), x = Convert.ToInt32(args[2]) /* age */;
	steps = Convert.ToInt32(args[3]);
    double[][] reserves = Outer(g, r, x);
	stopWatch.Stop();
	TimeSpan ts = stopWatch.Elapsed;
	Console.WriteLine("{0:00}.{1:00}", (ts.Hours*3600+ts.Minutes*60+ts.Seconds).ToString(), (ts.Milliseconds / 10));
	//Console.WriteLine("Execution time: {0:00}.{1:00}", (ts.Hours*3600+ts.Minutes*60+ts.Seconds).ToString(), (ts.Milliseconds / 10));
    // Hardcoded value calculated by Actulus
    //Print(new double[][] { new double[] { 0.61831213598263246 } });
    //Print(reserves);
  }
}


/*

  Bad t += h:                 Better t = b + (s - fullsteps) * h:

  0:    0.6147781089029630    0.6182286287542200
  1:    0.6389189039960340    0.6425841754258790
  2:    0.6636921825143570    0.6675866147160560
  3:    0.6890992850947200    0.6932376275473600
  4:    0.7151418398525670    0.7195385436806430
  5:    0.7418224202547390    0.7464906946562280
  6:    0.7691451636168510    0.7740958595299020
  7:    0.7971161884598430    0.8023568171620030
  8:    0.8257437339335060    0.8312780202150740
  9:    0.8550381401751990    0.8608664075148080
 10:    0.8850119886641060    0.8911323730449740
 11:    0.9156807752266860    0.9220909116005050
 12:    0.9470643296964210    0.9537629630448370
 13:    0.9791889214121570    0.9861769792513660
 14:    1.0120897860086300    1.0193707402161800
 15:    1.0458137979546600    1.0533934485955700
 16:    1.0804221631361100    1.0883081351644800
 17:    1.1159931861308700    1.1241944115712500
 18:    1.1526252699287100    1.1611516114943100
 19:    1.1904403127227700    1.1993023671767700
 20:    1.2295876231165700    1.2387966757073300
 21:    1.2702484338295100    1.2798165188372200
 22:    1.3126410788941100    1.3225811122387800
 23:    1.3570269089396900    1.3673528757977900
 24:    1.4037170433570900    1.4144442369328400
 25:    1.4530800899996400    1.4642254055509900
 26:    1.5055510013288000    1.5171332940390600
 27:    1.5616412829885400    1.5736818012208600
 28:    1.6219508307893900    1.6344737388468100
 29:    1.6871817500158500    1.7002147573495500
 30:    1.7581546129593400    1.7717297301363600
 31:    1.8327373818955700    1.8463745962513000
 32:    1.9071769924288800    1.9207480627254600
 33:    1.9804270393853200    1.9944107613526100
 34:    2.0510933586345000    2.0654101004882300
 35:    2.1173249431898300    2.1318997349996900
 36:    2.1766678633302200    2.1914494780362800
 37:    2.2258685075611400    2.2398650669836100
 38:    2.2606051795068300    2.2734172358768300
 39:    2.2751188554554400    2.2874842807873700
 40:    2.2617031584534000    2.2718033357283400
 41:    2.2277591495631200    2.2368587015675800
 42:    2.1893199805354400    2.1972868682017800
 43:    2.1464888231076700    2.1532319694754000
 44:    2.0993810601030400    2.1048741616004700
 45:    2.0481309769579400    2.0524286244565900
 46:    1.9929059282191700    1.9961440128275500
 47:    1.9339229094735100    1.9363003625603800
 48:    1.8714597417681200    1.8732064699298600
 49:    1.8058548759077500    1.8071967760958700
 50:    1.7374953558207200    1.7386278019940900
 51:    1.6667979879154900    1.6678741918133400
 52:    1.5941906816122800    1.5953244348860500
 53:    1.5200987314114100    1.5213763458647100
 54:    1.4449368839297700    1.4464323910296300
 55:    1.3691051808400600    1.3708949540867800
 56:    1.2929859530780300    1.2951616375719900
 57:    1.2169404185868300    1.2196206957898100
 58:    1.1413050540408300    1.1446466920120300
 59:    1.0663895969834500    1.0705964665057000
 60:    0.9924797704770350    0.9978054930430020
 61:    0.9198478219520920    0.9265846901767150
 62:    0.8487715092832020    0.8572177401545190
 63:    0.7795567909989170    0.7899589533625360
 64:    0.7125533330391370    0.7250317001054190
 65:    0.6481499967391670    0.6626274147129790
 66:    0.5867440511827810    0.6029051594693090
 67:    0.5286913037759830    0.5459917170122100
 68:    0.4742557257663510    0.4919821573269300
 69:    0.4235769502000460    0.4409407934657390
 70:    0.3766619250085270    0.3929023864123660
 71:    0.3333938236957770    0.3478733630885020
 72:    0.2935488543934290    0.3058326505738390
 73:    0.2568201575882660    0.2667315255240360
 74:    0.2228554907228520    0.2304918135938560
 75:    0.1913104238574920    0.1970023786055610
 76:    0.1619039622940920    0.1661160270488420
 77:    0.1344548161429540    0.1376533594430340
 78:    0.1088874788434810    0.1114252668523790
 79:    0.0852226298426998    0.0872860920493180
 80:    0.0635816199100679    0.0652165292069939
 81:    0.0442164963295822    0.0454065202929996
 82:    0.0275320887533484    0.0282799206899699
 83:    0.0140473060593912    0.0144160071889708
 84:    0.0043618568321764    0.0044682878951233
 85:    0.0000000000000000    0.0000000000000000
 */
