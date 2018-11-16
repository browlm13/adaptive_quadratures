# newton.py
#
# Daniel R. Reynolds
# Math5315 @ SMU
# Fall 2017

def newton(Ffun, Jfun, x, maxit=1000, Srtol=1e-15, Satol=1e-15, Rrtol=1e-15, Ratol=1e-15, output=False):
    """
    Usage: x, its = newton(Ffun, Jfun, x, maxit, Srtol, Satol, Rrtol, Ratol, output)

    This routine uses Newton's method to approximate a root of
    the nonlinear system of equations F(x)=0.  The iteration ceases 
    when the following condition is met:

       ||xnew - xold|| < atol + rtol*||xnew||

    inputs:   Ffun     nonlinear function name/handle
              Jfun     Jacobian function name/handle
              x        initial guess at solution
              maxit    maximum allowed number of iterations
              Srtol    relative solution tolerance
              Satol    absolute solution tolerance
              Rrtol    relative residual tolerance
              Ratol    absolute residual tolerance
              output   flag (true/false) to output iteration history/plot
    outputs:  x        approximate solution
              its      number of iterations taken
    """

    # imports
    import numpy as np
    import math
    import sys

    # check input arguments
    if (int(maxit) < 1):
        sys.stdout.write("newton: maxit = %i < 1. Resetting to 10\n" % (int(maxit)))
        maxit = 10
    if (Srtol < 1e-15):
        sys.stdout.write("newton: Srtol = %g < %g. Resetting to %g\n" % (Srtol, 1e-15, 1e-15))
        Srtol = 1e-15
    if (Satol < 0):
        sys.stdout.write("newton: Satol = %g < 0. Resetting to %g\n" % (Satol, 1e-15))
        Satol = 1e-15
    if (Rrtol < 1e-15):
        sys.stdout.write("newton: Rrtol = %g < %g. Resetting to %g\n" % (Rrtol, 1e-15, 1e-15))
        Rrtol = 1e-15
    if (Ratol < 0):
        sys.stdout.write("newton: Ratol = %g < 0. Resetting to %g\n" % (Ratol, 1e-15))
        Ratol = 1e-15

    # evaluate function
    F = Ffun(x)
    f0norm = np.linalg.norm(F)
        
    # perform iteration
    for its in range(1,maxit+1):

        # evaluate derivative
        J = Jfun(x)

        # compute Newton update, new guess at solution, new residual
        if (np.isscalar(x)):       # if problem is scalar-valued
            h = F/J
        else:                      # if problem is vector-valued
            h = np.linalg.solve(J, F)
        x = x - h
        F = Ffun(x)

        # check for convergence and output diagnostics
        hnorm = np.linalg.norm(h)
        xnorm = np.linalg.norm(x)
        fnorm = np.linalg.norm(F)
        if (output):
            sys.stdout.write("   iter %3i, \t||h|| = %g, \thtol = %g, \t||f|| = %g, \tftol = %g\n" % 
                             (its, hnorm, Satol+Srtol*xnorm, fnorm, Ratol+Rrtol*f0norm))

        if ( (hnorm < Satol + Srtol*xnorm) or (fnorm < Ratol + Rrtol*f0norm)):
            break

    return [x, its]

