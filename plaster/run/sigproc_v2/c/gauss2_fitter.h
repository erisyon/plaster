#ifndef GAUSS2_FITTER_H
#define GAUSS2_FITTER_H

typedef __uint8_t Uint8;
typedef __uint8_t np_uint8;

typedef __uint16_t Uint16;
typedef __uint16_t np_uint16;

typedef __uint32_t Uint32;
typedef __uint32_t np_uint32;

typedef __uint64_t Uint64;
typedef __uint64_t np_uint64;

typedef __uint128_t Uint128;
// Note. No numpy equivalent of 128

typedef __int8_t Sint8;
typedef __int8_t np_int8;

typedef __int16_t Sint16;
typedef __int16_t np_int16;

typedef __int32_t Sint32;
typedef __int32_t np_int32;

typedef __int64_t Sint64;
typedef __int64_t np_int64;

typedef __int128_t Sint128;
// Note. No numpy equivalent of 128

typedef float Float32;
typedef float np_float32;

typedef double Float64;
typedef double np_float64;

// In all of the following, the Gaussian parameters are in the order:
//    amplitude, sigma_x, sigma_y, pos_x, pos_y, rho, offset

#define N_INFO_ELEMENTS (10)
#define N_MAX_ITERATIONS (75)

// These must match in gauss2_fitter.py
#define PARAM_SIGNAL (0)
#define PARAM_NOISE (1)
#define PARAM_ASPECT_RATIO (2)
#define PARAM_FIRST_FIT_PARAM (3)
#define PARAM_AMP (3)
#define PARAM_SIGMA_Y (4)
#define PARAM_SIGMA_X (5)
#define PARAM_CENTER_Y (6)
#define PARAM_CENTER_X (7)
#define PARAM_RHO (8)
#define PARAM_OFFSET (9)
#define PARAM_LAST_FIT_PARAM (10)
#define PARAM_MEA (10)
#define PARAM_N_FULL_PARAMS (11)
#define PARAM_N_FIT_PARAMS (7)



void gauss_2d(double *p, double *dst_x, int m, int n, void *data);
void jac_gauss_2d(double *p, double *dst_jac, int m, int n, void *data);
char *get_dlevmar_stop_reason(int reason);


int fit_gauss_2d(
    np_float64 *pixels,
    np_int64 mea,
    np_float64 params[7],
    np_float64 *info,
    np_float64 *covar
);

int fit_gauss_2d_on_float_image(
    np_float32 *im,
    np_int64 im_h,
    np_int64 im_w,
    np_int64 center_y,
    np_int64 center_x,
    np_int64 mea,
    np_float64 params[7],
    np_float64 *info,
    np_float64 *covar,
    np_float64 *noise
);

int fit_array_of_gauss_2d_on_float_image(
    np_float32 *im,
    np_int64 im_h,
    np_int64 im_w,
    np_int64 mea,
    np_int64 n_peaks,
    np_int64 *center_y,
    np_int64 *center_x,
    np_float64 *params,
    np_int64 *fails
);

#endif
