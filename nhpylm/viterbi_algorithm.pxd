from nhpylm.npylm cimport NPYLM
cdef list viterbi_segment_sequences(NPYLM hpylm, list sequences_str)