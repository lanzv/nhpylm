from nhpylm.npylm cimport NPYLM
cdef void blocked_gibbs_iteration(NPYLM npylm, list sequences)