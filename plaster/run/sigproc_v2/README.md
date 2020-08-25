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
