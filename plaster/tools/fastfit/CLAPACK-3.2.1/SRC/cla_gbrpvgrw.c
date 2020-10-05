/* cla_gbrpvgrw.f -- translated by f2c (version 20061008).
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

doublereal cla_gbrpvgrw__(integer *n, integer *kl, integer *ku, integer *
	ncols, complex *ab, integer *ldab, complex *afb, integer *ldafb)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, afb_dim1, afb_offset, i__1, i__2, i__3, i__4;
    real ret_val, r__1, r__2, r__3;

    /* Builtin functions */
    double r_imag(complex *);

    /* Local variables */
    integer i__, j, kd;
    real amax, umax, rpvgrw;


/*     -- LAPACK routine (version 3.2.1)                                 -- */
/*     -- Contributed by James Demmel, Deaglan Halligan, Yozo Hida and -- */
/*     -- Jason Riedy of Univ. of California Berkeley.                 -- */
/*     -- April 2009                                                   -- */

/*     -- LAPACK is a software package provided by Univ. of Tennessee, -- */
/*     -- Univ. of California Berkeley and NAG Ltd.                    -- */

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CLA_GBRPVGRW computes the reciprocal pivot growth factor */
/*  norm(A)/norm(U). The "max absolute element" norm is used. If this is */
/*  much less than 1, the stability of the LU factorization of the */
/*  (equilibrated) matrix A could be poor. This also means that the */
/*  solution X, estimated condition numbers, and error bounds could be */
/*  unreliable. */

/*  Arguments */
/*  ========= */

/*     N       (input) INTEGER */
/*     The number of linear equations, i.e., the order of the */
/*     matrix A.  N >= 0. */

/*     KL      (input) INTEGER */
/*     The number of subdiagonals within the band of A.  KL >= 0. */

/*     KU      (input) INTEGER */
/*     The number of superdiagonals within the band of A.  KU >= 0. */

/*     NCOLS   (input) INTEGER */
/*     The number of columns of the matrix A.  NCOLS >= 0. */

/*     AB      (input) COMPLEX array, dimension (LDAB,N) */
/*     On entry, the matrix A in band storage, in rows 1 to KL+KU+1. */
/*     The j-th column of A is stored in the j-th column of the */
/*     array AB as follows: */
/*     AB(KU+1+i-j,j) = A(i,j) for max(1,j-KU)<=i<=min(N,j+kl) */

/*     LDAB    (input) INTEGER */
/*     The leading dimension of the array AB.  LDAB >= KL+KU+1. */

/*     AFB     (input) COMPLEX array, dimension (LDAFB,N) */
/*     Details of the LU factorization of the band matrix A, as */
/*     computed by CGBTRF.  U is stored as an upper triangular */
/*     band matrix with KL+KU superdiagonals in rows 1 to KL+KU+1, */
/*     and the multipliers used during the factorization are stored */
/*     in rows KL+KU+2 to 2*KL+KU+1. */

/*     LDAFB   (input) INTEGER */
/*     The leading dimension of the array AFB.  LDAFB >= 2*KL+KU+1. */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Statement Functions .. */
/*     .. */
/*     .. Statement Function Definitions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    afb_dim1 = *ldafb;
    afb_offset = 1 + afb_dim1;
    afb -= afb_offset;

    /* Function Body */
    rpvgrw = 1.f;
    kd = *ku + 1;
    i__1 = *ncols;
    for (j = 1; j <= i__1; ++j) {
	amax = 0.f;
	umax = 0.f;
/* Computing MAX */
	i__2 = j - *ku;
/* Computing MIN */
	i__4 = j + *kl;
	i__3 = min(i__4,*n);
	for (i__ = max(i__2,1); i__ <= i__3; ++i__) {
/* Computing MAX */
	    i__2 = kd + i__ - j + j * ab_dim1;
	    r__3 = (r__1 = ab[i__2].r, dabs(r__1)) + (r__2 = r_imag(&ab[kd +
		    i__ - j + j * ab_dim1]), dabs(r__2));
	    amax = dmax(r__3,amax);
	}
/* Computing MAX */
	i__3 = j - *ku;
	i__2 = j;
	for (i__ = max(i__3,1); i__ <= i__2; ++i__) {
/* Computing MAX */
	    i__3 = kd + i__ - j + j * afb_dim1;
	    r__3 = (r__1 = afb[i__3].r, dabs(r__1)) + (r__2 = r_imag(&afb[kd
		    + i__ - j + j * afb_dim1]), dabs(r__2));
	    umax = dmax(r__3,umax);
	}
	if (umax != 0.f) {
/* Computing MIN */
	    r__1 = amax / umax;
	    rpvgrw = dmin(r__1,rpvgrw);
	}
    }
    ret_val = rpvgrw;
    return ret_val;
} /* cla_gbrpvgrw__ */
