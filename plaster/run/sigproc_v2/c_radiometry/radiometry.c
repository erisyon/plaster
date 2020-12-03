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
#include "csa.h"

#define PI2 (2.0 * M_PI)


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


csa *_init_interpolate(
    Size n_samples,
    Float64 *x,
    Float64 *y,
    Float64 *val
) {
    point *points = (point *)alloca(sizeof(point) * n_samples);
    for(Index i=0; i<n_samples; i++) {
        point *p = &points[i];
        p->x = x[i];
        p->y = y[i];
        p->z = val[i];
    }

    csa *spline = csa_create();
    csa_addpoints(spline, n_samples, points);
    csa_calculatespline(spline);
    return spline;
}

Float64 _interpolate(csa *spline, Float64 x, Float64 y) {
    point p;
    p.x = x;
    p.y = y;
    p.z = 0.0;
    csa_approximatepoints(spline, 1, &p);
    return p.z;
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

//    trace("%f %f %f %f %f\n",
//        center_x,
//        center_y,
//        sigma_x,
//        sigma_y,
//        rho
//    );

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
    if(peak_i == 0) trace("loc(x,y) %f %f\n", loc_x, loc_y);
    ensure_only_in_debug(0 <= loc_x && loc_x < ctx->width, "loc_x out of bounds");
    ensure_only_in_debug(0 <= loc_y && loc_y < ctx->height, "loc_y out of bounds");

    // corner is the lower left pixel coorinate in image coordinates
    // where the (mea, mea) sub-image will be extracted
    // Add 0.5 to round up as opposed to floor to keep the spots more centered
    Index corner_x = floor(loc_x - half_mea + 0.5);
    Index corner_y = floor(loc_y - half_mea + 0.5);
    if(peak_i == 0) trace("corner(x,y) %ld %ld\n", corner_x, corner_y);
    ensure_only_in_debug(0 <= corner_x && corner_x < ctx->width, "corner_x out of bounds");
    ensure_only_in_debug(0 <= corner_y && corner_y < ctx->height, "corner_y out of bounds");

    // center is the location relative to the the corner
    Float64 center_x = loc_x - corner_x;
    Float64 center_y = loc_y - corner_y;
    if(peak_i == 0) trace("center(x,y) %f %f\n", center_x, center_y);
    ensure_only_in_debug(0 <= center_x && center_x < mea, "center out of bounds");
    ensure_only_in_debug(0 <= center_y && center_y < mea, "center out of bounds");

    // Shape
    Index n_divs_minus_one = ctx->n_divs - 1;

    if(peak_i == 0) trace("n_divs %f   raw(w,h) %f %f\n", ctx->n_divs, ctx->raw_width, ctx->raw_height);

    Float64 *psf_pixels = (Float64 *)alloca(sizeof(Float64) * mea_sq);
    Float64 *dat_pixels = (Float64 *)alloca(sizeof(Float64) * mea_sq);

    Index ch_i = 0;
    for(Index cy_i=0; cy_i<n_cycles; cy_i++) {
        Float64 focus = *f64arr_ptr1(&ctx->focus_adjustment, cy_i);

        Float64 sigma_x = _interpolate(ctx->_interp_sigma_x, loc_x, loc_y);
        Float64 sigma_y = _interpolate(ctx->_interp_sigma_y, loc_x, loc_y);
        Float64 rho = _interpolate(ctx->_interp_rho, loc_x, loc_y);

        sigma_x *= focus;
        sigma_y *= focus;

        if(peak_i == 0) trace("loc(x,y) %f %f  cen(x,y) %f %f  foc %f  adj_sig(x,y) %f %f  rho %f\n",
            loc_x, loc_y,
            center_x, center_y,
            focus,
            sigma_x, sigma_y, rho
        );

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

//        if(peak_i == 0) {
//            trace("cy_i %ld\n", cy_i);
//            _dump_vec(psf_pixels, mea, mea, "psf_pixels");
//            _dump_vec(dat_pixels, mea, mea, "dat_pixels");
//        }

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
//        if(peak_i == 0) trace("sig %f  psf_sum_square %f\n", signal, psf_sum_square);

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
//        trace("res_mean %f\n", residual_mean);

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
//        trace("res_var %f\n", residual_var);

        // NOISE
        Float64 noise = sqrt(residual_var / psf_sum_square);
//        trace("noise %f\n", noise);

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


char *context_init(RadiometryContext *ctx) {
    int n_samples = ctx->n_reg_psf_samples;
    int n_divs = 6;
//    trace("n_samples=%ld\n", n_samples);
//    _dump_vec(f64arr_ptr1(&ctx->reg_psf_x, 0), n_divs, n_divs, "x");
//    _dump_vec(f64arr_ptr1(&ctx->reg_psf_y, 0), n_divs, n_divs, "y");
//    _dump_vec(f64arr_ptr1(&ctx->reg_psf_sigma_x, 0), n_divs, n_divs, "reg_psf_sigma_x");
//    _dump_vec(f64arr_ptr1(&ctx->reg_psf_sigma_y, 0), n_divs, n_divs, "reg_psf_sigma_y");
//    _dump_vec(f64arr_ptr1(&ctx->reg_psf_rho, 0), n_divs, n_divs, "reg_psf_rho");

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
    return NULL;
}


char *context_free(RadiometryContext *ctx) {
    csa_destroy(ctx->_interp_sigma_x);
    csa_destroy(ctx->_interp_sigma_y);
    csa_destroy(ctx->_interp_rho);
    return NULL;
}
