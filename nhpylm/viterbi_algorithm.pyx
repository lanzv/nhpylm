from nhpylm.npylm cimport NPYLM
from nhpylm.sequence cimport EOS, BOS
import numpy as np
cimport numpy as np
np.import_array()
DTYPE = np.float64


cdef list viterbi_segment_sequences(NPYLM npylm, list sequences_str):
    """
    Process viterbi segmentation for all testing sequences based on the hpylm model.

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    sequence : list of strings
        list of current training sequences we are sampling, their segmentation will be changed
    Returns
    -------
    final_segmetation : list of lists of strings
        list of segmented sequences, one sequence segmentation is represented as list of strigns(~segments)
    """
    cdef list final_segmentation = []
    for sequence_string in sequences_str:
        final_segmentation.append(__get_sequence_segmentation(npylm, sequence_string))
    return final_segmentation



cdef list __get_sequence_segmentation(NPYLM npylm, str sequence_str):
    """
    Process the forward filtering to get precomputed alpha array.
    With the alpha values, do the backward sampling to predict the segmentation.

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    sequence_str : string
        current sequence we are sampling, its segmentation will be changed after this function
    Returns
    -------
    final_segmetation : list of lists of strings
        list of segmented sequences, one sequence segmentation is represented as list of strigns(~segments)
    """
    cdef np.ndarray alpha
    cdef dict bigram_cache_p
    alpha, bigram_cache_p = __forward_filtering(npylm, sequence_str)
    return __get_segments_with_backward_sampling(npylm, sequence_str, alpha, bigram_cache_p)


cdef tuple __forward_filtering(NPYLM npylm, str sequence_str):
    """
    Process the forward filtering and precompute alpha array with marginalized probabilities
    for all t, k (t is the position in sequence, k is the length of the last segment).
    The scaling is used for avoiding of underflowing. The original paper used expsumlog().

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    sequence_str : string
        current sequence we are sampling, its segmentation will be changed after this function
    Returns
    -------
    alpha : np.ndarray
        precompute alpha[t,k] array with marginalized probabilities for all t, k 
        (t is the position in sequence, k is the length of the last segment)
    bigram_cache_p : dict of dicts of floats
        dictionary of cached probabilities of bigrams in order to decrease number of 
        get_bigram_probability calls, first key is the second gram, second key is a first gram
    """
    cdef int sequence_len = len(sequence_str)
    cdef int max_segment_size = npylm.max_segment_size
    cdef np.ndarray alpha = np.zeros([sequence_len+1, max_segment_size+1], dtype=DTYPE)
    cdef dict bigram_cache_p = {} # first dictionary keys are second grams, second - inner - dictionary keys are first grams
    cdef float prob
    cdef np.ndarray scaling_alpha = np.zeros([sequence_len+1], dtype=DTYPE)
    cdef int t, k, j
    cdef float sum_prob
    cdef float prod_scaling
    cdef float sum_alpha_t

    alpha[0, 0] = 1.0
    for t in range(1, sequence_len+1):
        prod_scaling = 1.0
        sum_alpha_t = 0.0
        for k in range(1, min(max_segment_size, t)+1):
            if k != 1:
                prod_scaling *= scaling_alpha[t-k+1]
            sum_prob = 0.0
            if t-k == 0:
                # first gram is an <bos> beggining of sequence
                # Cache probabilities to avoid of still calling get_bigram_probability function
                if sequence_str[t-k:t] in bigram_cache_p and BOS in bigram_cache_p[sequence_str[t-k:t]]:
                    prob = bigram_cache_p[sequence_str[t-k:t]][BOS]
                else:
                    # second gram: (t-k+1):(t) (by the "vector indexing")
                    prob = npylm.get_bigram_probability(BOS, sequence_str[t-k:t])
                    if (not sequence_str[t-k:t] in bigram_cache_p) or (not BOS in bigram_cache_p[sequence_str[t-k:t]]):
                        bigram_cache_p[sequence_str[t-k:t]] = {}
                    bigram_cache_p[sequence_str[t-k:t]][BOS] = prob

                sum_prob += (prob * alpha[0, 0])
            else:
                # for j in range(1, t-k) in the original word segmentation paper - we have to consider max_size
                for j in range(1, min(max_segment_size, t-k)+1): 
                    # Cache probabilities to avoid of still calling get_bigram_probability function
                    if sequence_str[t-k:t] in bigram_cache_p and sequence_str[t-k-j:t-k] in bigram_cache_p[sequence_str[t-k:t]]:
                        prob = bigram_cache_p[sequence_str[t-k:t]][sequence_str[t-k-j:t-k]]
                    else:
                        # first gram: (t-k-j+1):(t-k), second gram: (t-k+1):(t) (by the "vector indexing")
                        prob = npylm.get_bigram_probability(sequence_str[t-k-j:t-k], sequence_str[t-k:t])
                        if (not sequence_str[t-k:t] in bigram_cache_p) or (not sequence_str[t-k-j:t-k] in bigram_cache_p[sequence_str[t-k:t]]):
                            bigram_cache_p[sequence_str[t-k:t]] = {}
                        bigram_cache_p[sequence_str[t-k:t]][sequence_str[t-k-j:t-k]] = prob
                    sum_prob += (prob * alpha[t-k, j])

            alpha[t, k] = sum_prob*prod_scaling
            sum_alpha_t += sum_prob*prod_scaling

        # Perform scaling to avoid underflowing
        for k in range(1, min(max_segment_size, t)+1):
            alpha[t, k] /= sum_alpha_t
        scaling_alpha[t] = 1.0/sum_alpha_t

    return alpha, bigram_cache_p


cdef list __get_segments_with_backward_sampling(NPYLM npylm, str sequence_str, np.ndarray alpha, dict bigram_cache_p):
    """
    Get segmentation using the backward sampling regarding the precompued alpha.
    The final segmentation is chosen
    based on argmax of k ~ p(w_{i} | c_{t-k+1}^{t}, Theta) * alpha[t][k] in each step.

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    sequence_str : string
        current sequence we are sampling, its segmentation will be changed after this function
    alpha : np.ndarray
        precompute alpha[t,k] array with marginalized probabilities for all t, k 
        (t is the position in sequence, k is the length of the last segment)
    bigram_cache_p : dict of dicts of floats
        dictionary of cached probabilities of bigrams in order to decrease number of 
        get_bigram_probability calls, first key is the second gram, second key is a first gram
    Returns
    -------
    final_segmetation : list of lists of strings
        list of segmented sequences, one sequence segmentation is represented as list of strigns(~segments)
    """
    cdef int t = len(sequence_str)
    cdef int max_segment_size = npylm.max_segment_size
    cdef list segmented_sequence = []
    cdef int i, j, k
    cdef list probs
    cdef float prob
    cdef list k_candidates
    cdef list borders = [t]
    cdef str w = EOS


    while t > 0:   
        probs = []
        k_candidates = []
        for k in range(1, min(max_segment_size, t)+1):
            k_candidates.append(k)
            # Use bigram prob cache if possible
            if w in bigram_cache_p and sequence_str[t-k:t] in bigram_cache_p[w]:
                prob = bigram_cache_p[w][sequence_str[t-k:t]]
            else:
                # first gram: (t-k+1):(k), second gram: w (by the "vector indexing")
                prob = npylm.get_bigram_probability(sequence_str[t-k:t], w)
            probs.append(prob * alpha[t, k])
        k = k_candidates[np.argmax(probs)]
        w = sequence_str[t-k:t]
        t -= k
        borders.append(t)


    borders.reverse()
    for i, j in zip(borders[:-1], borders[1:]):
        segmented_sequence.append(sequence_str[i:j])

    return segmented_sequence