from nhpylm.npylm cimport NPYLM, SHPYLMNode, CHPYLMNode
import numpy as np
cimport numpy as np
cdef void apply_hyperparameters_learning(NPYLM npylm, list train_sequences, bint d_theta_learning, bint poisson_learning)
cdef void update_poisson_lambda(NPYLM npylm, list train_sequences)
cdef void update_poisson_k_probs(NPYLM npylm, int segment_samples = *)



# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXX          Segment Hierarchical Pitman-Yor Tree Hyperparameters          XXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

cdef void update_shpylm_d_theta(NPYLM npylm)
cdef void __recursive_shpylm_d_theta_preparation(SHPYLMNode node, np.ndarray sum1_minus_y_ui, 
                                                np.ndarray sum1_minus_z_uwkj, np.ndarray sumy_ui,
                                                np.ndarray sumlogx_u)
cdef float __get_shpylm_1_minus_y_ui(SHPYLMNode node)
cdef float __get_shpylm_1_minus_z_uwkj_sum(SHPYLMNode node)
cdef float __get_shpylm_y_ui_sum(SHPYLMNode node)
cdef float __get_shpylm_logx_u(SHPYLMNode node)




# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXX         Character Hierarchical Pitman-Yor Tree Hyperparameters        XXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

cdef void update_chpylm_d_theta(NPYLM npylm)
cdef void update_chpylm_d_theta(NPYLM npylm)
cdef void __recursive_chpylm_d_theta_preparation(CHPYLMNode node, np.ndarray sum1_minus_y_ui, 
                                                np.ndarray sum1_minus_z_uwkj, np.ndarray sumy_ui,
                                                np.ndarray sumlogx_u)
cdef float __get_chpylm_1_minus_y_ui(CHPYLMNode node)
cdef float __get_chpylm_1_minus_z_uwkj_sum(CHPYLMNode node)
cdef float __get_chpylm_y_ui_sum(CHPYLMNode node)
cdef float __get_chpylm_logx_u(CHPYLMNode node)