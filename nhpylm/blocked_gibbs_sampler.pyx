from nhpylm.npylm cimport NPYLM
from nhpylm.sequence cimport Sequence, EOS, BOS
from nhpylm.random_utils cimport random_choice
import numpy as np
cimport numpy as np
np.import_array()
DTYPE = np.float64


cdef void blocked_gibbs_iteration(NPYLM npylm, list sequences):
    """
    Process one blocked gibbs iteration for all training sequences.
    First shuffle sequences randomly. Sample each sequence - remove sequence segments
    from NPYLM, get new segmentation, add new sequence segments to NPYLM.
    At the end of all sequence sampling also sample hyperparameters.

    Parameters
    ----------
    hpylm : NPYLM
        current state of hpylm language model
    sequence : list
        list of current training sequences we are sampling, their segmentation will be changed
    """
    cdef int sequence_id
    cdef np.ndarray rand_indices = np.arange(len(sequences))

    np.random.shuffle(rand_indices)
    for sequence_id in rand_indices:
        npylm.remove_sequence(sequences[sequence_id])
        __optmize_sequence_segmentation(npylm, sequences[sequence_id])
        npylm.add_sequence(sequences[sequence_id])

cdef void __optmize_sequence_segmentation(NPYLM npylm, Sequence sequence):
    """
    Process the forward filtering to get precomputed alpha array.
    With the alpha values, do the backward sampling to create a new segmentation.

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    sequence : Sequence
        current sequence we are sampling, its segmentation will be changed after this function
    """

    cdef np.ndarray alpha
    cdef dict bigram_cache_p
    alpha, bigram_cache_p = __forward_filtering(npylm, sequence)
    __backward_sampling(npylm, sequence, alpha, bigram_cache_p)


cdef tuple __forward_filtering(NPYLM npylm, Sequence sequence):
    """
    Process the forward filtering and precompute alpha array with marginalized probabilities
    for all t, k (t is the position in sequence, k is the length of the last segment).
    The scaling is used for avoiding of underflowing. The original paper used expsumlog().

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    sequence : Sequence
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
    cdef int sequence_len = len(sequence.sequence_string)
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
                if sequence.sequence_string[t-k:t] in bigram_cache_p and BOS in bigram_cache_p[sequence.sequence_string[t-k:t]]:
                    prob = bigram_cache_p[sequence.sequence_string[t-k:t]][BOS]
                else:
                    # second gram: (t-k+1):(t) (by the "vector indexing")
                    prob = npylm.get_bigram_probability(BOS, sequence.sequence_string[t-k:t])
                    if (not sequence.sequence_string[t-k:t] in bigram_cache_p) or (not BOS in bigram_cache_p[sequence.sequence_string[t-k:t]]):
                        bigram_cache_p[sequence.sequence_string[t-k:t]] = {}
                    bigram_cache_p[sequence.sequence_string[t-k:t]][BOS] = prob

                sum_prob += (prob * alpha[0, 0])
            else:
                # for j in range(1, t-k) in the original word segmentation paper - we have to consider max_size
                for j in range(1, min(max_segment_size, t-k)+1): 
                    # Cache probabilities to avoid of still calling get_bigram_probability function
                    if sequence.sequence_string[t-k:t] in bigram_cache_p and sequence.sequence_string[t-k-j:t-k] in bigram_cache_p[sequence.sequence_string[t-k:t]]:
                        prob = bigram_cache_p[sequence.sequence_string[t-k:t]][sequence.sequence_string[t-k-j:t-k]]
                    else:
                        # first gram: (t-k-j+1):(t-k), second gram: (t-k+1):(t) (by the "vector indexing")
                        prob = npylm.get_bigram_probability(sequence.sequence_string[t-k-j:t-k], sequence.sequence_string[t-k:t])
                        if (not sequence.sequence_string[t-k:t] in bigram_cache_p) or (not sequence.sequence_string[t-k-j:t-k] in bigram_cache_p[sequence.sequence_string[t-k:t]]):
                            bigram_cache_p[sequence.sequence_string[t-k:t]] = {}
                        bigram_cache_p[sequence.sequence_string[t-k:t]][sequence.sequence_string[t-k-j:t-k]] = prob
                    sum_prob += (prob * alpha[t-k, j])

            alpha[t, k] = sum_prob*prod_scaling
            sum_alpha_t += sum_prob*prod_scaling

        # Perform scaling to avoid underflowing
        for k in range(1, min(max_segment_size, t)+1):
            alpha[t, k] /= sum_alpha_t
        scaling_alpha[t] = 1.0/sum_alpha_t

    return alpha, bigram_cache_p


cdef void __backward_sampling(NPYLM npylm, Sequence sequence, np.ndarray alpha, dict bigram_cache_p):
    """
    Do the backward sampling to get optimized segmentation with np.random.choice of 
    all k candidates k ~ p(w_{i} | c_{t-k+1}^{t}, Theta) * alpha[t][k] in each step.

    Parameters
    ----------
    npylm : NPYLM
        current state of npylm language model
    sequence : Sequence
        current sequence we are sampling, its segmentation will be changed after this function
    alpha : np.ndarray
        precompute alpha[t,k] array with marginalized probabilities for all t, k 
        (t is the position in sequence, k is the length of the last segment).
    bigram_cache_p : dict of dicts of floats
        dictionary of cached probabilities of bigrams in order to decrease number of 
        get_bigram_probability calls, first key is the second gram, second key is a first gram
    """
    cdef int t = len(sequence.sequence_string)
    cdef int max_segment_size = npylm.max_segment_size
    cdef int k
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
            if w in bigram_cache_p and sequence.sequence_string[t-k:t] in bigram_cache_p[w]:
                prob = bigram_cache_p[w][sequence.sequence_string[t-k:t]]
            else:
                # first gram: (t-k+1):(k), second gram: w (by the "vector indexing")
                prob = npylm.get_bigram_probability(sequence.sequence_string[t-k:t], w)
            probs.append(prob * alpha[t, k])

        k = k_candidates[random_choice(probs)]
        w = sequence.sequence_string[t-k:t]
        t -= k
        borders.append(t)

    
    borders.reverse()
    sequence.set_segmentation(borders)