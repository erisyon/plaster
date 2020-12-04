#include "stdint.h"
#include "alloca.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdarg.h"
#include "memory.h"
#include "pthread.h"
#include "unistd.h"
#include "math.h"
#include "c_common.h"
#include "_radiometry.h"

#define PI2 (2.0 * M_PI)


/*
    SPLINE INTERPOLATION NOTES

    At some point it would be nice to get this spline interpolator working
    but after a few hours I gave up and switched to a higher-res sampling
    of the PSF form the python spline

    //csa *_init_interpolate(
    //    Size n_samples,
    //    Float64 *x,
    //    Float64 *y,
    //    Float64 *val
    //) {
    //    point *points = (point *)alloca(sizeof(point) * n_samples);
    //    for(Index i=0; i<n_samples; i++) {
    //        point *p = &points[i];
    //        p->x = x[i];
    //        p->y = y[i];
    //        p->z = val[i];
    //    }
    //
    //    csa *spline = csa_create();
    //    csa_addpoints(spline, n_samples, points);
    //    csa_calculatespline(spline);
    //    return spline;
    //}
    //
    //Float64 _interpolate(csa *spline, Float64 x, Float64 y) {
    //    point p;
    //    p.x = x;
    //    p.y = y;
    //    p.z = 0.0;
    //    csa_approximatepoints(spline, 1, &p);
    //    return p.z;
    //}
*/

void _dump_vec(Float64 *vec, int width, int height, char *msg) {
    trace("VEC %s [\n", msg);
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            fprintf(_log, "%4.4f, ", vec[y*width + x]);
        }
        fprintf(_log, "\n");
    }
    fprintf(_log, "]\n");
    fflush(_log);
}

void psf_im(
    Float64 center_x, Float64 center_y,
    Float64 sigma_x, Float64 sigma_y,
    Float64 rho, Float64 *pixels, Size mea
) {
    center_x -= 0.5;
    center_y -= 0.5;

    Float64 sgxs = sigma_x * sigma_x;
    Float64 sgys = sigma_y * sigma_y;
    Float64 rs = rho * rho;
    Float64 omrs = 1.0 - rs;
    Float64 tem_a = 1.0 / (sigma_x * sigma_y * omrs);
    Float64 denom = 2.0 * (rho - 1.0) * (rho + 1.0) * sgxs * sgys;
    Float64 numer_const = -2.0 * rho * sigma_x * sigma_y;
    Float64 linear_term = tem_a * sqrt(omrs);

    Float64 *dst = pixels;
    for (int i=0; i<mea; i++) {
        Float64 y = (Float64)i;
        Float64 ympy = y - center_y;
        for (int j=0; j<mea; j++) {
            Float64 x = (Float64)j;
            Float64 xmpx = x - center_x;
            *dst++ = (
                linear_term * exp(
                    (
                        numer_const * xmpx * ympy
                        + sgxs * ympy * ympy
                        + sgys * xmpx * xmpx
                    ) / denom
                ) / PI2
            );
        }
    }
}


Float64 *_get_psf_at_loc(RadiometryContext *ctx, Float64 loc_x, Float64 loc_y) {
    Index x_i = floor(ctx->n_divs * loc_x / ctx->width);
    Index y_i = floor(ctx->n_divs * loc_y / ctx->height);
    ensure_only_in_debug(0 <= x_i && x_i < ctx->width, "loc x out of bounds");
    ensure_only_in_debug(0 <= y_i && y_i < ctx->height, "loc x out of bounds");
    return f64arr_ptr2(&ctx->reg_psf_samples, y_i, x_i);
}


char *radiometry_field_stack_one_peak(RadiometryContext *ctx, Index peak_i) {
    /*
    Each cycle is sub-pixel aligned, but each peak can be at
    an arbitrary fractional offset (which has already been determine
    by the caller).

    Each cycle can also have a focus correction which applies to the
    sigmas of every peak in that cycle.

    Each region has gauss params (sigma_x, sigma_y, rho) which are adjusted
    by the focus correction.

    Thus, every peak, every cycle has its own Gaussian parameters.

    Possible future optimizations:
        * Realistically there's probably only 0.1 of a pixel of precision
          in the position, sigmas, and rho so we could pre-compute these
          PSF images and just look them up instead of rebuilding them at every
          peak/channel/cycle/
    */

    Float64 *im = f64arr_ptr2(&ctx->chcy_ims, 0, 0);

    // Position
    ensure(ctx->n_channels == 1, "Only 1 channel supported until I have a chance to implement it");
    Size n_cycles = ctx->n_cycles;
    Size mea = ctx->peak_mea;
    Size mea_sq = mea * mea;
    Float64 half_mea = (Float64)mea / 2.0;

    // loc is the location in image coordinates
    Float64 *loc_p = f64arr_ptr1(&ctx->locs, peak_i);
    Float64 loc_x = loc_p[1];
    Float64 loc_y = loc_p[0];
    ensure_only_in_debug(0 <= loc_x && loc_x < ctx->width, "loc_x out of bounds");
    ensure_only_in_debug(0 <= loc_y && loc_y < ctx->height, "loc_y out of bounds");

    // corner is the lower left pixel coorinate in image coordinates
    // where the (mea, mea) sub-image will be extracted
    // Add 0.5 to round up as opposed to floor to keep the spots more centered
    Index corner_x = floor(loc_x - half_mea + 0.5);
    Index corner_y = floor(loc_y - half_mea + 0.5);
    ensure_only_in_debug(0 <= corner_x && corner_x < ctx->width, "corner_x out of bounds");
    ensure_only_in_debug(0 <= corner_y && corner_y < ctx->height, "corner_y out of bounds");

    // center is the location relative to the the corner
    Float64 center_x = loc_x - corner_x;
    Float64 center_y = loc_y - corner_y;
    ensure_only_in_debug(0 <= center_x && center_x < mea, "center out of bounds");
    ensure_only_in_debug(0 <= center_y && center_y < mea, "center out of bounds");

    // Shape
    Index n_divs_minus_one = ctx->n_divs - 1;

    Float64 *psf_pixels = (Float64 *)alloca(sizeof(Float64) * mea_sq);
    Float64 *dat_pixels = (Float64 *)alloca(sizeof(Float64) * mea_sq);

    Index ch_i = 0;
    for(Index cy_i=0; cy_i<n_cycles; cy_i++) {
        Float64 focus = *f64arr_ptr1(&ctx->focus_adjustment, cy_i);

        Float64 *psf_params = _get_psf_at_loc(ctx, loc_x, loc_y);
        Float64 sigma_x = psf_params[0];
        Float64 sigma_y = psf_params[1];
        Float64 rho = psf_params[2];

        sigma_x *= focus;
        sigma_y *= focus;

        psf_im(
            center_x, center_y,
            sigma_x, sigma_y,
            rho, psf_pixels, ctx->peak_mea
        );

        // COPY the data into a contiguous buffer
        Float64 *dst_p = dat_pixels;
        for(Index y=0; y<mea; y++) {
            Float64 *dat_p = f64arr_ptr4(&ctx->chcy_ims, ch_i, cy_i, corner_y+y, corner_x);
            for(Index x=0; x<mea; x++) {
                *dst_p++ = *dat_p++;
            }
        }

        // SIGNAL
        Float64 psf_sum_square = 0.0;
        Float64 signal = 0.0;
        Float64 *psf_p = psf_pixels;
        Float64 *dat_p = dat_pixels;
        for(Index i=0; i<mea_sq; i++) {
            signal += *psf_p * *dat_p;
            psf_sum_square += *psf_p * *psf_p;
            psf_p ++;
            dat_p ++;
        }
        signal /= psf_sum_square;

        // RESIDUALS mean
        Float64 residual_mean = 0.0;
        psf_p = psf_pixels;
        dat_p = dat_pixels;
        for(Index i=0; i<mea_sq; i++) {
            Float64 residual = *dat_p - signal * *psf_p;
            residual_mean += residual;
            psf_p ++;
            dat_p ++;
        }
        residual_mean /= (Float64)mea_sq;

        // RESIDUALS variance
        Float64 residual_var = 0.0;
        psf_p = psf_pixels;
        dat_p = dat_pixels;
        for(Index i=0; i<mea_sq; i++) {
            Float64 residual = *dat_p - signal * *psf_p;
            Float64 mean_centered = residual - residual_mean;
            residual_var += mean_centered * mean_centered;
            psf_p ++;
            dat_p ++;
        }
        residual_var /= (Float64)mea_sq;

        // NOISE
        Float64 noise = sqrt(residual_var / psf_sum_square);

        // SNR
        Float64 snr = signal / noise;

        // ASPECT-RATIO
        // TODO
        Float64 aspect_ratio = 1.0;

        Index ch_i = 0;
        Float64 *out = f64arr_ptr3(&ctx->out_radiometry, peak_i, ch_i, cy_i);
        out[0] = signal;
        out[1] = noise;
        out[2] = snr;
        out[3] = aspect_ratio;
    }

    return NULL;
}


char *test_interp(RadiometryContext *ctx, Float64 loc_x, Float64 loc_y, Float64 *out_vals) {
    Float64 *psf_params = _get_psf_at_loc(ctx, loc_x, loc_y);
    trace("%f %f %f\n", loc_x, loc_y, psf_params[0]);
    Float64 sigma_x = psf_params[0];
    Float64 sigma_y = psf_params[1];
    Float64 rho = psf_params[2];
    out_vals[0] = sigma_x;
    out_vals[1] = sigma_y;
    out_vals[2] = rho;
    return NULL;
}

char *context_init(RadiometryContext *ctx) {
    /*
    See SPLINE INTERPOLATION NOTES

    int n_samples = ctx->n_reg_psf_samples;
    int n_divs = ctx->n_divs;
    trace("n_samples=%ld\n", n_samples);
    _dump_vec(f64arr_ptr1(&ctx->reg_psf_x, 0), n_divs, n_divs, "x");
    _dump_vec(f64arr_ptr1(&ctx->reg_psf_y, 0), n_divs, n_divs, "y");
    _dump_vec(f64arr_ptr1(&ctx->reg_psf_sigma_x, 0), n_divs, n_divs, "reg_psf_sigma_x");
    _dump_vec(f64arr_ptr1(&ctx->reg_psf_sigma_y, 0), n_divs, n_divs, "reg_psf_sigma_y");
    _dump_vec(f64arr_ptr1(&ctx->reg_psf_rho, 0), n_divs, n_divs, "reg_psf_rho");

    ctx->_interp_sigma_x = _init_interpolate(
        ctx->n_reg_psf_samples,
        f64arr_ptr1(&ctx->reg_psf_x, 0),
        f64arr_ptr1(&ctx->reg_psf_y, 0),
        f64arr_ptr1(&ctx->reg_psf_sigma_x, 0)
    );

    ctx->_interp_sigma_y = _init_interpolate(
        ctx->n_reg_psf_samples,
        f64arr_ptr1(&ctx->reg_psf_x, 0),
        f64arr_ptr1(&ctx->reg_psf_y, 0),
        f64arr_ptr1(&ctx->reg_psf_sigma_y, 0)
    );

    ctx->_interp_rho = _init_interpolate(
        ctx->n_reg_psf_samples,
        f64arr_ptr1(&ctx->reg_psf_x, 0),
        f64arr_ptr1(&ctx->reg_psf_y, 0),
        f64arr_ptr1(&ctx->reg_psf_rho, 0)
    );
    */

    return NULL;
}


char *context_free(RadiometryContext *ctx) {
    /*
    See SPLINE INTERPOLATION NOTES

    csa_destroy(ctx->_interp_sigma_x);
    csa_destroy(ctx->_interp_sigma_y);
    csa_destroy(ctx->_interp_rho);
    */
    return NULL;
}
