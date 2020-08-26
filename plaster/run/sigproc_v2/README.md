# TODO

* Research and test the z-stack issue.
    Is the focus metric bad? /  biased?
    Why are all z-stacks seemingly pushed to the edge the same amount
    even though by eye they seem more centered?

* Research the PSF extraction code: Why does the background make
the PSFs more peaky around the edges?
    - Experimental evidence:
        When we put in a perfect background im the problem
        went away.
        When I put in a a perfect estimate of the regional background
        the problem comes back
    - Theory: The interp function needs to be centered on the divs
        That is, it uses the lower left corner as the anchor but the
        calculation is an average over the whole div.
    - I'm prettu sure it needs a 2d spline extrapolator
        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html#scipy.interpolate.RectBivariateSpline
        See: https://stackoverflow.com/questions/34053174/python-scipy-for-2d-extrapolated-spline-function

* Move the explore_synth code from notebook into
    zest_sigproc_v2_integration


* Get tests to pass: Last run fails
=====  . zest_sigproc_v2_integration ===================================================================================================================
File .../zest/zest.py:498 in function do
    func()
File ../plaster/plaster/run/sigproc_v2/zests/zest_sigproc_v2_integration.py:27 in function zest_sigproc_v2_integration
    raise NotImplementedError
raised: NotImplementedError

===== zest_compute_channel_weights . it_returns_balanced_channels ======================================================================================
File .../zest/zest.py:498 in function do
    func()
File ../plaster/plaster/run/sigproc_v2/zests/zest_sigproc_v2_worker.py:433 in function it_returns_balanced_channels
    balance = worker._analyze_step_1a_compute_channel_weights(sigproc_params)
raised: TypeError
_analyze_step_1a_compute_channel_weights() missing 1 required positional argument: 'calib'

===== zest_import_balanced_images . it_balances_regionally =============================================================================================
File .../zest/zest.py:498 in function do
    func()
File ../plaster/plaster/run/sigproc_v2/zests/zest_sigproc_v2_worker.py:510 in function it_balances_regionally
    balanced_ims = worker._analyze_step_1_import_balanced_images(
raised: TypeError
_analyze_step_1_import_balanced_images() missing 1 required positional argument: 'calib'

===== zest_import_balanced_images . it_remaps_and_balances_channels ====================================================================================
File .../zest/zest.py:498 in function do
    func()
File ../plaster/plaster/run/sigproc_v2/zests/zest_sigproc_v2_worker.py:472 in function it_remaps_and_balances_channels
    balanced_ims = worker._analyze_step_1_import_balanced_images(
raised: TypeError
_analyze_step_1_import_balanced_images() missing 1 required positional argument: 'calib'

===== zest_peak_radiometry . it_nans_negative_signal_or_noise ==========================================================================================
File .../zest/zest.py:498 in function do
    func()
File ../plaster/plaster/run/sigproc_v2/zests/zest_sigproc_v2_worker.py:693 in function it_nans_negative_signal_or_noise
    signal, noise = worker._analyze_step_5a_peak_radiometry(
raised: AttributeError
module 'plaster.run.sigproc_v2.sigproc_v2_worker' has no attribute '_analyze_step_5a_peak_radiometry'

===== zest_peak_radiometry . it_sub_pixel_alignments_no_noise ==========================================================================================
File .../zest/zest.py:498 in function do
    func()
File ../plaster/plaster/run/sigproc_v2/zests/zest_sigproc_v2_worker.py:671 in function it_sub_pixel_alignments_no_noise
    signal, noise = worker._analyze_step_5a_peak_radiometry(
raised: AttributeError
module 'plaster.run.sigproc_v2.sigproc_v2_worker' has no attribute '_analyze_step_5a_peak_radiometry'

===== zest_peak_radiometry . it_gets_the_residuals =====================================================================================================
File .../zest/zest.py:498 in function do
    func()
File ../plaster/plaster/run/sigproc_v2/zests/zest_sigproc_v2_worker.py:682 in function it_gets_the_residuals
    signal, noise = worker._analyze_step_5a_peak_radiometry(
raised: AttributeError
module 'plaster.run.sigproc_v2.sigproc_v2_worker' has no attribute '_analyze_step_5a_peak_radiometry'

===== zest_peak_radiometry . it_gets_a_perfect_result_with_no_noise_and_perfect_alignment ==============================================================
File .../zest/zest.py:498 in function do
    func()
File ../plaster/plaster/run/sigproc_v2/zests/zest_sigproc_v2_worker.py:661 in function it_gets_a_perfect_result_with_no_noise_and_perfect_alignment
    signal, noise = worker._analyze_step_5a_peak_radiometry(
raised: AttributeError
module 'plaster.run.sigproc_v2.sigproc_v2_worker' has no attribute '_analyze_step_5a_peak_radiometry'