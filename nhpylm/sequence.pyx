cdef str BOS = "<bos>" # beginning of sequence
cdef str EOS = "<eos>" # end of sequence
cdef str BOU = "<bou>" # beginning of unit
cdef str EOU = "<eou>" # end of unit

cdef class Sequence():
    def __cinit__(self, str sequence_string):
        """
        Constructor of Sequence data structure.

        Parameters
        ----------
        sequence_string : string
            sequence string in its very basic form (no prefixes/suffixes/..)
        """
        self.sequence_string = sequence_string
        self.segmentation = [sequence_string]


    cdef void set_segmentation(self, list borders):
        """
        Function that get list of borders of segmetns and it will reupload sequence's segmentation
        regarding those borders.

        example:
            borders=[0, 3, 4, 10], sequence_string="abcdefghij" -> segmentation=["abc", "d", "efghij"]

        Parameters
        ----------
        borders : list of ints
            list of indices of sequence_string string chars (in other word char array),
            that describe segment borders - they have to be order by ascending
        """
        cdef list segmented_sequence = []
        cdef int i, j
        for i, j in zip(borders[:-1], borders[1:]):
            segmented_sequence.append(self.sequence_string[i:j])
        self.segmentation = segmented_sequence