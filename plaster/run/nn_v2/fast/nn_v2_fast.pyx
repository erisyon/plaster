import time
cimport c_nn_v2_fast as c_nn
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from libc.stdlib cimport calloc, free


def test_nn(test_nn_params, prep_result, sim_v2_result):
    cdef c_nn.DyeType [:, ::1] dyetracks_view
    ? cdef csim.Uint8 [:, ::1] dyetracks_view

    shape = sim_v2_result.test_radmat.shape
    assert len(shape) == 4
    n_rows = shape[0] * shape[1]
    n_cols = shape[2] * shape[3]

    radmat = sim_v2_result.test_radmat.reshape((n_rows, n_cols))

    dyetracks_view =


    cdef c_nn.Context ctx
    ctx.n_cols = n_cols
    ctx.radmat_n_rows = n_rows
    ctx.radmat;
    ctx.train_dyetracks_n_rows;
    ctx.train_dyetracks;
    ctx.train_dyepeps_n_rows;
    ctx.train_dyepeps;
    ctx.output_callrecs;


        c_nn.context_start()
