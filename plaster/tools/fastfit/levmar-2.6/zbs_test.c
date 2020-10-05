#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

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


void gauss1(double *p, double *x, double *e, int m, int n, void *data) {
    register int i;

    double mu = p[0];
    double sigma = p[1];
    double sigma2 = sigma * sigma;

    for(i=0; i<n; ++i) {
        x[i] = exp( -(i-mu) / (2.0 * sigma2) );
    }
}

/*
void jac_gauss1(double *p, double *jac, int m, int n, void *data) {
    register int i, j;

    for(i=j=0; i<n; ++i) {
        jac[j++]=(-2 + 2 * p[0] - 4 * ROSD * (p[1] - p[0] * p[0]) *p[0]);
        jac[j++]=(2 * ROSD * (p[1] - p[0] * p[0]));
    }
}
*/



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

    /*
    // Rosenbrock function
    double p[5], x[16];
    m=2;
    n=2;
    p[0]=-1.2;
    p[1]=1.0;
    for(i=0; i<n; i++) {
        x[i]=0.0;
    }
    ret = dlevmar_der(ros, jacros, p, x, m, n, 1000, opts, info, NULL, NULL, NULL); // with analytic Jacobian
    */


    // Gauss1 function
    double x[] = { 3.7e-06, 3.4e-04, 1.1e-02, 1.4e-01, 6.1e-01, 1.0e+00, 6.1e-01, 1.4e-01, 1.1e-02, 3.4e-04 };
    m = 2; // n_params
    n = 10; // len of data

    double p[] = { 4.5, 1.2 };

    ret = dlevmar_der(

        gauss1,
        NULL,//jac_gauss1,
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




