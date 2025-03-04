cdef str BOS
cdef str EOS
cdef str BOU
cdef str EOU
cdef class Sequence():
    cdef str sequence_string
    cdef list segmentation
    cdef void set_segmentation(self, list borders)