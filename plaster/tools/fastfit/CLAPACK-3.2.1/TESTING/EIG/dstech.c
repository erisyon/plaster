/* dstech.f -- translated by f2c (version 20061008).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"
#include "blaswrap.h"

/* Subroutine */ int dstech_(integer *n, doublereal *a, doublereal *b,
	doublereal *eig, doublereal *tol, doublereal *work, integer *info)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2, d__3;

    /* Local variables */
    integer i__, j;
    doublereal mx, eps, emin;
    integer isub, bpnt, numl, numu, tpnt, count;
    doublereal lower, upper, tuppr;
    extern doublereal dlamch_(char *);
    extern /* Subroutine */ int dstect_(integer *, doublereal *, doublereal *,
	     doublereal *, integer *);
    doublereal unflep;


/*  -- LAPACK test routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*     Let T be the tridiagonal matrix with diagonal entries A(1) ,..., */
/*     A(N) and offdiagonal entries B(1) ,..., B(N-1)).  DSTECH checks to */
/*     see if EIG(1) ,..., EIG(N) are indeed accurate eigenvalues of T. */
/*     It does this by expanding each EIG(I) into an interval */
/*     [SVD(I) - EPS, SVD(I) + EPS], merging overlapping intervals if */
/*     any, and using Sturm sequences to count and verify whether each */
/*     resulting interval has the correct number of eigenvalues (using */
/*     DSTECT).  Here EPS = TOL*MAZHEPS*MAXEIG, where MACHEPS is the */
/*     machine precision and MAXEIG is the absolute value of the largest */
/*     eigenvalue. If each interval contains the correct number of */
/*     eigenvalues, INFO = 0 is returned, otherwise INFO is the index of */
/*     the first eigenvalue in the first bad interval. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The dimension of the tridiagonal matrix T. */

/*  A       (input) DOUBLE PRECISION array, dimension (N) */
/*          The diagonal entries of the tridiagonal matrix T. */

/*  B       (input) DOUBLE PRECISION array, dimension (N-1) */
/*          The offdiagonal entries of the tridiagonal matrix T. */

/*  EIG     (input) DOUBLE PRECISION array, dimension (N) */
/*          The purported eigenvalues to be checked. */

/*  TOL     (input) DOUBLE PRECISION */
/*          Error tolerance for checking, a multiple of the */
/*          machine precision. */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension (N) */

/*  INFO    (output) INTEGER */
/*          0  if the eigenvalues are all correct (to within */
/*             1 +- TOL*MAZHEPS*MAXEIG) */
/*          >0 if the interval containing the INFO-th eigenvalue */
/*             contains the incorrect number of eigenvalues. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Check input parameters */

    /* Parameter adjustments */
    --work;
    --eig;
    --b;
    --a;

    /* Function Body */
    *info = 0;
    if (*n == 0) {
	return 0;
    }
    if (*n < 0) {
	*info = -1;
	return 0;
    }
    if (*tol < 0.) {
	*info = -5;
	return 0;
    }

/*     Get machine constants */

    eps = dlamch_("Epsilon") * dlamch_("Base");
    unflep = dlamch_("Safe minimum") / eps;
    eps = *tol * eps;

/*     Compute maximum absolute eigenvalue, error tolerance */

    mx = abs(eig[1]);
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
/* Computing MAX */
	d__2 = mx, d__3 = (d__1 = eig[i__], abs(d__1));
	mx = max(d__2,d__3);
/* L10: */
    }
/* Computing MAX */
    d__1 = eps * mx;
    eps = max(d__1,unflep);

/*     Sort eigenvalues from EIG into WORK */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	work[i__] = eig[i__];
/* L20: */
    }
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	isub = 1;
	emin = work[1];
	i__2 = *n + 1 - i__;
	for (j = 2; j <= i__2; ++j) {
	    if (work[j] < emin) {
		isub = j;
		emin = work[j];
	    }
/* L30: */
	}
	if (isub != *n + 1 - i__) {
	    work[isub] = work[*n + 1 - i__];
	    work[*n + 1 - i__] = emin;
	}
/* L40: */
    }

/*     TPNT points to singular value at right endpoint of interval */
/*     BPNT points to singular value at left  endpoint of interval */

    tpnt = 1;
    bpnt = 1;

/*     Begin loop over all intervals */

L50:
    upper = work[tpnt] + eps;
    lower = work[bpnt] - eps;

/*     Begin loop merging overlapping intervals */

L60:
    if (bpnt == *n) {
	goto L70;
    }
    tuppr = work[bpnt + 1] + eps;
    if (tuppr < lower) {
	goto L70;
    }

/*     Merge */

    ++bpnt;
    lower = work[bpnt] - eps;
    goto L60;
L70:

/*     Count singular values in interval [ LOWER, UPPER ] */

    dstect_(n, &a[1], &b[1], &lower, &numl);
    dstect_(n, &a[1], &b[1], &upper, &numu);
    count = numu - numl;
    if (count != bpnt - tpnt + 1) {

/*        Wrong number of singular values in interval */

	*info = tpnt;
	goto L80;
    }
    tpnt = bpnt + 1;
    bpnt = tpnt;
    if (tpnt <= *n) {
	goto L50;
    }
L80:
    return 0;

/*     End of DSTECH */

} /* dstech_ */
