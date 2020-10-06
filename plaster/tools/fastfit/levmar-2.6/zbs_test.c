#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "assert.h"

#include "levmar.h"

#ifndef LM_DBL_PREC
#error Demo program assumes that levmar has been compiled with double precision, see LM_DBL_PREC!
#endif


/* Sample functions to be minimized with LM and their Jacobians.
 * More test functions at http://www.csit.fsu.edu/~burkardt/f_src/test_nls/test_nls.html
 * Check also the CUTE problems collection at ftp://ftp.numerical.rl.ac.uk/pub/cute/;
 * CUTE is searchable through http://numawww.mathematik.tu-darmstadt.de:8081/opti/select.html
 * CUTE problems can also be solved through the AMPL web interface at http://www.ampl.com/TRYAMPL/startup.html
 *
 * Nonlinear optimization models in AMPL can be found at http://www.princeton.edu/~rvdb/ampl/nlmodels/
 */

/* Rosenbrock function, global minimum at (1, 1) */
/*
#define ROSD 105.0

void ros(double *p, double *x, int m, int n, void *data) {
    register int i;
    for(i=0; i<n; ++i) {
        x[i]=(
            (1.0 - p[0])*(1.0 - p[0]) + ROSD*(
                p[1] - p[0] * p[0]
            ) * (
                p[1] - p[0] * p[0])
        );
    }
}

void jacros(double *p, double *jac, int m, int n, void *data) {
    register int i, j;

    for(i=j=0; i<n; ++i) {
        jac[j++]=(-2 + 2 * p[0] - 4 * ROSD * (p[1] - p[0] * p[0]) *p[0]);
        jac[j++]=(2 * ROSD * (p[1] - p[0] * p[0]));
    }
}
*/

void gauss_1d(double *p, double *dst_x, double *e, int m, int n, void *data) {
    // p = parameters array [a, b, c]
    // dst_x = Where to write to
    // e = error terms
    // m = number of parameters
    // n = number of data points
    // data = data

    double a = p[0];
    double b = p[1];
    double c = p[2];
    double _2c2 = 2.0 * c * c;

    for(int i=0; i<n; i++) {
        double x = (double)i - b;
        dst_x[i] = a * exp( -(x*x) / _2c2 );
    }
}


void jac_gauss_1d(double *p, double *jac, int m, int n, void *data) {
    // p = parameters array [a, b, c]
    // jac = Where to write to
    // m = number of parameters
    // n = number of data points
    // data = data

    // d/da = exp( -(x - b)**2 / 2 * c**2 )
    // d/db = ( a * (x - b) * exp( -(x-b)**2 / 2c**2 ) ) / c**2
    // d/dc = ( a * (x - b)**2 * exp( -(x-b)**2 / 2c**2 ) ) / c**3

    double a = p[0];
    double b = p[1];
    double c = p[2];

    double _c2 = c * c;
    double _c3 = _c2 * c;
    double _2c2 = 2.0 * c * c;

    int j = 0;
    for(int i=0; i<n; i++) {
        double x = (double)i;
        double _xb = x - b;
        double _xb2 = _xb * _xb;
        double _exp1 = exp( -_xb2 / _2c2 );
        double _axb = a * _xb;
        double _axb_exp1 = _axb * _exp1;
        jac[j++] = _exp1;
        jac[j++] = _axb_exp1 / _c2;
        jac[j++] = _axb_exp1 * _xb / _c3;
    }
}


int main() {
    register int i;//, j;
    int ret;
    int m, n;

    double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
    opts[0]=LM_INIT_MU;
    opts[1]=1E-15;
    opts[2]=1E-15;
    opts[3]=1E-20;
    opts[4]= LM_DIFF_DELTA; // relevant only if the Jacobian is approximated using finite differences; specifies forward differencing
    //opts[4]=-LM_DIFF_DELTA; // specifies central differencing to approximate Jacobian; more accurate but more expensive to compute!

    // Gauss1 function
    // True params are: 1.0, 25.0, 5.0
    double x[] = { 1.2e-02,  1.2e-02,  5.5e-03,  1.7e-02, -3.0e-03,  7.0e-03,
       -1.7e-02,  4.1e-03, -1.8e-04, -1.0e-06, -4.1e-03,  1.7e-02,
        2.1e-02,  6.6e-02,  6.5e-02,  1.3e-01,  2.0e-01,  2.8e-01,
        3.6e-01,  5.0e-01,  6.0e-01,  7.4e-01,  8.4e-01,  9.2e-01,
        9.7e-01,  9.9e-01,  9.9e-01,  9.4e-01,  8.4e-01,  7.4e-01,
        6.0e-01,  4.9e-01,  3.6e-01,  2.8e-01,  2.0e-01,  1.4e-01,
        9.7e-02,  5.8e-02,  4.3e-02,  3.8e-02,  3.1e-02,  1.7e-02,
       -3.3e-03, -1.1e-04, -7.7e-03,  2.2e-02,  9.0e-03,  2.6e-04,
       -2.6e-03, -1.0e-02
    };

    m = 3; // n_params
    n = 50; // len of data

    double p[] = { 0.9, 20.0, 4.0 };

    ret = dlevmar_der(
        gauss_1d,
        jac_gauss_1d,
        p,
        x,
        m,
        n,
        1000,  // max_iter
        opts,
        info,
        NULL,
        NULL,
        NULL
    ); // with analytic Jacobian

    /*
    ret = dlevmar_dif(
        gauss1,
        p,
        x,
        m,
        n,
        1000,  // max_iter
        opts,
        info,
        NULL,
        NULL,
        NULL
    ); // without analytic Jacobian
    */

    printf("Levenberg-Marquardt returned %d in %g iter, reason %g\nSolution: ", ret, info[5], info[6]);
    for(i=0; i<m; ++i) {
        printf("%.7g ", p[i]);
    }
    printf("\n\nMinimization info:\n");
    for(i=0; i<LM_INFO_SZ; ++i) {
        printf("%g ", info[i]);
    }
    printf("\n");
    return 0;
}




