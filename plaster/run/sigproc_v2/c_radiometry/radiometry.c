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

void psf_im(
    Float64 center_x, Float64 center_y,
    Float64 sigma_x, Float64 sigma_y,
    Float64 rho, Float64 *pixels, Size mea
) {
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

    // Position
    ensure(ctx->n_channels == 1, "Only 1 channel supported until I have a chance to implement it");
    Size n_cycles = ctx->n_cycles;
    Size mea = ctx->peak_mea;
    Size mea_sq = mea * mea;
    Float64 *loc_p = f64arr_ptr1(&ctx->locs, peak_i);
    Float64 half_mea = (Float64)mea / 2.0;
    Float64 loc_x = loc_p[1];
    Float64 loc_y = loc_p[0];
    Float64 center_x = half_mea + loc_p[1] - floor(loc_p[1]);
    Float64 center_y = half_mea + loc_p[0] - floor(loc_p[0]);
    Index loc_lft = (Index)floor(loc_x - half_mea);

    // Shape
    Index n_divs_minus_one = ctx->n_divs - 1;
    Index reg_y = min(n_divs_minus_one, max(0, ctx->n_divs * loc_y / ctx->raw_height));
    Index reg_x = min(n_divs_minus_one, max(0, ctx->n_divs * loc_x / ctx->raw_width));
    Float64 *reg_psf_params_p = f64arr_ptr2(&ctx->reg_psf_params, reg_y, reg_x);
    Float64 sigma_x = reg_psf_params_p[0];
    Float64 sigma_y = reg_psf_params_p[1];
    Float64 rho = reg_psf_params_p[2];

    Float64 *psf_pixels = (Float64 *)alloca(sizeof(Float64) * mea_sq);
    Float64 *dat_pixels = (Float64 *)alloca(sizeof(Float64) * mea_sq);

    for(Index cy_i=0; cy_i<n_cycles; cy_i++) {
        Float64 focus = *f64arr_ptr1(&ctx->focus_adjustment, cy_i);

        psf_im(
            center_x, center_y,
            sigma_x * focus, sigma_y * focus,
            rho, psf_pixels, ctx->peak_mea
        );

        // COPY the data into a contiguous buffer
        Float64 *dst_p = dat_pixels;
        for(Index y=0; y<mea; y++) {
            Float64 *dat_p = f64arr_ptr2(&ctx->chcy_ims, y, loc_lft);
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


char *context_init(RadiometryContext *ctx) {
    return NULL;
}


char *context_free(RadiometryContext *ctx) {
    return NULL;
}
