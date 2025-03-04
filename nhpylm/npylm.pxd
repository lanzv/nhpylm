from nhpylm.sequence cimport Sequence
import numpy as np
cimport numpy as np

cdef class NPYLM():
    cdef int max_segment_size
    cdef float beta_stops, beta_passes
    cdef int shpylm_max_depth, chpylm_max_depth
    cdef list shpylm_ds, shpylm_thetas, chpylm_ds, chpylm_thetas
    cdef float init_d, init_theta, d_a, d_b, theta_alpha, theta_beta
    cdef float theta
    cdef float poisson_lambda
    cdef float init_poisson_a
    cdef float init_poisson_b
    cdef list poisson_k_probs
    cdef set character_vocabulary
    cdef dict root_tables_context_lengths
    cdef int last_removed_shpylm_table_index
    cdef CHPYLMNode chpylm_root # root of character HPYLM tree
    cdef SHPYLMNode shpylm_root # root of segment HPYLM tree
    cdef void add_sequence(self, Sequence sequence)
    cdef void remove_sequence(self, Sequence sequence)
    cdef float get_bigram_probability(self, str first_gram, str second_gram)
    cdef float get_segmentation_log_probability(self, list sequence_segmentation)
    cdef void add_segment_to_chpylm(self, str segment)
    cdef void remove_segment_from_chpylm(self, str segment)
    cdef float get_G0_probability(self, str segment)



# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXX          Segment Hierarchical Pitman-Yor Tree Node          XXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

cdef class SHPYLMNode():
    cdef int depth
    cdef SHPYLMNode parent
    cdef NPYLM npylm
    cdef str hpylm_type # 't' for character hpylm, 's' for shant hpylm
    cdef dict children # dictionary of deeper context HPYLNodes
    cdef str context # one previous element
    cdef dict tables # dict of all tables - key: segment, value: Tables
    cdef dict c_wh # c(w|h) .. count of all customers 
    cdef int c_h # sum of all counts in c(w|h)
    cdef int t_h # t_{h} .. sum of counts of tables in tables dictionary
    cdef bint add_segment(self, str segment, list context, float pwhcomma)
    cdef bint remove_segment(self, str segment, list context)
    cdef float get_pwh_probability(self, str segment, list context, float pwhcomma)


cdef class STables():
    # t_hw could be get as len(self.tables)
    cdef list tables # list of ints - customer counts
    cdef str segment
    cdef SHPYLMNode hpylmnode
    cdef bint add_customer(self, float pwhcomma, int t_h)
    cdef bint remove_customer(self)


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXX         Character Hierarchical Pitman-Yor Tree Node        XXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

cdef class CHPYLMNode():
    cdef CHPYLMNode parent
    cdef NPYLM npylm
    cdef int depth
    cdef int stops # stop counts
    cdef int passes # pass counts
    cdef str hpylm_type # 't' for character hpylm, 's' for shant hpylm
    cdef dict children # dictionary of deeper context HPYLNodes
    cdef str context # one previous element
    cdef dict tables # dict of all tables - key: segment, value: Tables
    cdef dict c_wh # c(w|h) .. count of all customers 
    cdef int c_h # sum of all counts in c(w|h)
    cdef int t_h # t_{h} .. sum of counts of tables in tables dictionary
    cdef bint add_character(self, str character, list context, float pwhcomma)
    cdef bint remove_character(self, str character, list context)
    cdef float get_pwh_probability(self, str character, list context, float pwhcomma, float prev_pass_product)
    cdef int sample_context_length(self, str character, list context, float pwhcomma)
    cdef void __fill_sample_prob_table(self, list sample_prob_table, str character, list context, float pwhcomma, float prev_pass_product)

cdef class CTables():
    # t_hw could be get as len(self.tables)
    cdef list tables # list of ints - customer counts
    cdef str character
    cdef CHPYLMNode hpylmnode
    cdef bint add_customer(self, float pwhcomma, int t_h)
    cdef bint remove_customer(self)