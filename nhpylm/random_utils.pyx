from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport exp, pow

cdef int random_choice(list probabilities):
    """
    Do the random choice regarding the probabilities distribution.
    Return the random index of coming list corresponding to the list's distribution.
    The probabilities list doesn't have to sum up to 1, the function computes it itself.
    In case that sum of probabilities values sum up to 0, the uniform distribution is used.

    Parameters
    ----------
    probabilities : list of floats
        list of probabilities, doesn't have to sum up to 1
    Returns
    -------
    i : int
        chosen index of the random choice function regarding the probabilities distribution
    """
    cdef int i
    cdef float p
    cdef float random_number = float(rand())/float(RAND_MAX)
    cdef float prob_sum = 0.0
    cdef float left_border = 0.0
    cdef float normalized_p

    for p in probabilities:
        prob_sum += p
    if prob_sum == 0:
        # Uniform distribution
        for i in range(len(probabilities)):
            normalized_p = 1.0/len(probabilities)
            if random_number <= left_border + normalized_p:
                return i
            left_border += normalized_p
    else:
        # Given probabilities distribution
        for i, p in enumerate(probabilities):
            normalized_p = p/prob_sum
            if random_number <= left_border + normalized_p:
                return i
            left_border += normalized_p

cdef float poisson(int k, float lam):
    """
    Poisson distribution. 
    Po(k, lambda) = exp(-lambda)*((lambda^k)/(k!))

    Parameters
    ----------
    k : int
        size k
    lam : float
        lambda
    Returns
    -------
    poisson : float
        value get from the poisson distribution
    """
    cdef int k_factorial
    cdef int i
    if k == 0:
        k_factorial = 1
    else:
        k_factorial = 1
        for i in range(1, k+1):
            k_factorial *= i

    poisson = exp(-lam)*((pow(lam, k))/(k_factorial))

    return poisson

cdef int bernoulli(float prob):
    """
    Bernoulli distribution. 
    Return 1 with probability prob, return 0 with probability 1-prob.

    Parameters
    ----------
    prob : float
        probability of sampling 1
    Returns
    -------
    bernoulli : int
        sampled 1 or 0, depending on the prob
    """
    cdef float random_number = float(rand())/float(RAND_MAX)
    if random_number < prob:
        return 1
    else:
        return 0
