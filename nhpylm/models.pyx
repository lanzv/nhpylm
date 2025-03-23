from nhpylm.viterbi_algorithm cimport viterbi_segment_sequences
from nhpylm.blocked_gibbs_sampler cimport blocked_gibbs_iteration
from nhpylm.hyperparameters cimport apply_hyperparameters_learning
from nhpylm.npylm cimport NPYLM
from nhpylm.sequence cimport Sequence
from libc.math cimport exp
cimport numpy as np
import numpy as np
import logging


cdef class NHPYLMModel:
    cdef int max_segment_size
    cdef int n_gram
    cdef float init_d
    cdef float init_theta
    cdef float init_a
    cdef float init_b
    cdef float beta_stops
    cdef float beta_passes
    cdef float d_a
    cdef float d_b
    cdef float theta_alpha
    cdef float theta_beta
    cdef dict train_statistics, dev_statistics
    cdef NPYLM npylm

    
    def __init__(self, max_segment_size = 7, n_gram = 2,
                init_d = 0.5, init_theta = 2.0,
                init_a = 6.0, init_b = 0.8333,
                beta_stops = 1.0, beta_passes = 1.0,
                d_a = 1.0, d_b = 1.0, theta_alpha = 1.0, theta_beta = 1.0):
        """
        Init Nested Hirearchical Pitman-Yor Language Model for Sequences
        
        Parameters
        ----------
        max_segment_size : int
            maximum segment size that the model considers
        n_gram : int
            the n gram of word hirearchical pitman your language model, for now we only support bigrams
        init_d : float
            initial discount factor in G ~ PY(G0, d, theta) - used in probability recursive computation
            hyperparameter d is learned during training
        init_theta : float
            initial theta that controls the average similarity of G and G0 in G ~ PY(G0, d, theta) - used in probability recrusive computation
            hyperparameter theta is learned during trainign
        init_a : float
            initial 'a' argument in gamma distribution G(a, b) in order to estimate lambda for poisson correction
        init_b : float
            initial 'b' argument in gamma distribution G(a, b) in order to estimate lambda for poisson correction
        beta_stops : float
            stops hyperparameter, that will be used during sampling and probability computation in CHPYLM
            that helps us avoid the situation of zero stops in depth that we want to include
        beta_passes : float
            stops hyperparameter, that will be used during sampling and probability computation in CHPYLM
            that helps us to avoid the situation of zero passes in depth that we want to include
        d_a : float
            a parameter for d learning as a base of alpha of Beta distribution
        d_b : float
            b parameter for d learning as a base of beta of Beta distribution
        theta_alpha : float
            alpha parameter for theta learning as a base of shape of Gamma distribution
        theta_beta : float
            beta parameter for theta learning as base of scale of Gamma distribution
        """
        self.max_segment_size = max_segment_size
        self.n_gram = n_gram
        if not n_gram == 2:
            raise NotImplementedError("For now, we support only bigrams, but {} gram was given.".format(n_gram))
        self.init_d = init_d
        self.init_theta = init_theta
        self.init_a = init_a
        self.init_b = init_b
        self.beta_stops = beta_stops
        self.beta_passes = beta_passes
        self.d_a = d_a
        self.d_b = d_b
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        logging.info("The NHPYLM model was initialized with max segment size {}".format(max_segment_size))


    cpdef void train(self, list train_data, list dev_data, int epochs, bint d_theta_learning, 
                    bint poisson_learning, int print_each_nth_iteration = 5):
        """
        Perform the training process of the NHPYLM model. Each iteration print current statistics.
        Apply the hyperparameter learning after each iteration. Function parameters specify those learnings.

        Parameters
        ----------
        train_data : list of strings
            list of training sequences, each represented as a string
        dev_data : list of strings
            list of dev sequences, each represented as a string
        epochs : int
            number of training epochs
        d_theta_learning : boolean
            whether we want to apply d theta learning after each epoch or not
        poisson_learning : boolean
            whether we want to apply poisson learning or not
        print_each_nth_iteration : int
            print only iterations modulo print_each_nth_iteration
        """
        cdef int i
        cdef str sequence_str
        cdef list train_sequences = []
        cdef list dev_sequences = []
        cdef set train_character_vocabulary = set()
        cdef str character
        cdef list train_segments, dev_segments
        cdef float train_perplexity, dev_perplexity
        cdef Sequence sequence
        logging.info("NHPYLM train - {} train sequences, {} dev sequences.".format(len(train_data), len(dev_data)))

        # Prepare Sequences
        for sequence_str in train_data:
            train_sequences.append(Sequence(sequence_str))
            for character in sequence_str:
                train_character_vocabulary.add(character)
        for sequence_str in dev_data:
            dev_sequences.append(Sequence(sequence_str))

        # Initialize NPYLM and load all training sequences to it
        self.npylm = NPYLM(self.max_segment_size, self.init_d, self.init_theta, 
                            self.init_a, self.init_b,
                            train_character_vocabulary,
                            self.beta_stops, self.beta_passes,
                            self.d_a, self.d_b, self.theta_alpha, self.theta_beta)
        for sequence in train_sequences:
            self.npylm.add_sequence(sequence)

        # Training
        for i in range(epochs):
            blocked_gibbs_iteration(self.npylm, train_sequences)
            apply_hyperparameters_learning(self.npylm, train_sequences, d_theta_learning, poisson_learning)

            if (i+1)%print_each_nth_iteration == 0:
                train_segments, train_perplexity = self.predict_segments(train_data)
                dev_segments, dev_perplexity = self.predict_segments(dev_data)
                logging.info("\n{}. iteration\n\tTrain Perplexity: {}\n\tDev Perplexity: {}".format(i+1, train_perplexity, dev_perplexity)) # TODO REPLACE BY PRINT


    cpdef tuple predict_segments(self, list sequences):
        """
        Segment sequences using this trained model. Compute perplexity.

        Parameters
        ----------
        sequences : list of strings
            list of training sequences, each represented as a string
        Returns
        -------
        segmented_sequences : list of lists of strings
            list of sequences, each sequence is represented as list of strings
        perplexity : float
            compute perplexity of segmented sequences
        """
        cdef float perplexity
        cdef float prob_sum = 0.0
        cdef list sequence_segmentation
        cdef list segmented_sequences = viterbi_segment_sequences(self.npylm, sequences)
        for sequence_segmentation in segmented_sequences:
            prob_sum += self.npylm.get_segmentation_log_probability(sequence_segmentation)/len(sequence_segmentation)
        perplexity = exp(-prob_sum/len(segmented_sequences))
        return segmented_sequences, perplexity


cdef class NHPYLMClassesModel:
    cdef int max_segment_size
    cdef int n_gram
    cdef float init_d
    cdef float init_theta
    cdef float init_a
    cdef float init_b
    cdef float beta_stops
    cdef float beta_passes
    cdef float d_a
    cdef float d_b
    cdef float theta_alpha
    cdef float theta_beta
    cdef dict train_statistics, dev_statistics
    cdef dict npylm_classes

    
    def __init__(self, max_segment_size = 7, n_gram = 2,
                init_d = 0.5, init_theta = 2.0,
                init_a = 6.0, init_b = 0.8333,
                beta_stops = 1.0, beta_passes = 1.0,
                d_a = 1.0, d_b = 1.0, theta_alpha = 1.0, theta_beta = 1.0):
        """
        Init Nested Hirearchical Pitman-Yor Language Model for Gregorian Sequences
        
        Parameters
        ----------
        max_segment_size : int
            maximum segment size that the model considers
        n_gram : int
            the n gram of word hirearchical pitman your language model, for now we only support bigrams
        init_d : float
            initial discount factor in G ~ PY(G0, d, theta) - used in probability recursive computation
            hyperparameter d is learned during training
        init_theta : float
            initial theta that controls the average similarity of G and G0 in G ~ PY(G0, d, theta) - used in probability recrusive computation
            hyperparameter theta is learned during trainign
        init_a : float
            initial 'a' argument in gamma distribution G(a, b) in order to estimate lambda for poisson correction
        init_b : float
            initial 'b' argument in gamma distribution G(a, b) in order to estimate lambda for poisson correction
        beta_stops : float
            stops hyperparameter, that will be used during sampling and probability computation in CHPYLM
            that helps us avoid the situation of zero stops in depth that we want to include
        beta_passes : float
            stops hyperparameter, that will be used during sampling and probability computation in CHPYLM
            that helps us to avoid the situation of zero passes in depth that we want to include
        d_a : float
            a parameter for d learning as a base of alpha of Beta distribution
        d_b : float
            b parameter for d learning as a base of beta of Beta distribution
        theta_alpha : float
            alpha parameter for theta learning as a base of shape of Gamma distribution
        theta_beta : float
            beta parameter for theta learning as base of scale of Gamma distribution
        """
        self.max_segment_size = max_segment_size
        self.n_gram = n_gram
        if not n_gram == 2:
            raise NotImplementedError("For now, we support only bigrams, but {} gram was given.".format(n_gram))
        self.init_d = init_d
        self.init_theta = init_theta
        self.init_a = init_a
        self.init_b = init_b
        self.beta_stops = beta_stops
        self.beta_passes = beta_passes
        self.d_a = d_a
        self.d_b = d_b
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        logging.info("The NHPYLM model was initialized with max segment size {}".format(max_segment_size))


    cpdef void train(self, list train_data, list dev_data, list train_labels, list dev_labels, 
                    int epochs, bint d_theta_learning, bint poisson_learning, int print_each_nth_iteration = 5):
        """
        Initialize eight NHPYLMs, one nhpylm model for each class. Divide data and train NHPYLMs
        Apply the hyperparameter learning after each iteration. Function parameters specify those learnings.

        Parameters
        ----------
        train_data : list of strings
            list of training sequences, each represented as a string
        dev_data : list of strings
            list of dev sequences, each represented as a string
        train_labels : list of strings
            list of train labels
        dev_labels : list of strings
            list of dev labels
        epochs : int
            number of training epochs
        d_theta_learning : boolean
            whether we want to apply d theta learning after each epoch or not
        poisson_learning : boolean
            whether we want to apply poisson learning or not
        print_each_nth_iteration : int
            print only iterations modulo print_each_nth_iteration
        """
        cdef int i
        cdef str sequence_str
        cdef list train_sequences = []
        cdef list dev_sequences = []
        cdef set train_character_vocabulary = set()
        cdef str character
        cdef list train_segments, dev_segments
        cdef float train_perplexity, dev_perplexity
        cdef Sequence sequence
        cdef str cl
        cdef set classes_set
        cdef list classes
        cdef float correct, train_accuracy, dev_accuracy
        cdef str gold_y, pred_y
        logging.info("NHPYLM train - {} train sequences, {} dev sequences.".format(len(train_data), len(dev_data)))

        # Prepare Sequences
        for sequence_str in train_data:
            train_sequences.append(Sequence(sequence_str))
            for character in sequence_str:
                train_character_vocabulary.add(character)
        for sequence_str in dev_data:
            dev_sequences.append(Sequence(sequence_str))

        # Initialize NPYLM and load all training sequences to it
        self.npylm_classes = {}
        classes_set = set()
        for cl in train_labels:
            classes_set.add(cl)
        classes = list(classes_set)
        for cl in classes:
            self.npylm_classes[cl] = NPYLM(self.max_segment_size, self.init_d, self.init_theta, 
                            self.init_a, self.init_b,
                            train_character_vocabulary,
                            self.beta_stops, self.beta_passes,
                            self.d_a, self.d_b, self.theta_alpha, self.theta_beta)
        cdef dict train_sequences_classes = {}
        cdef NPYLM npylm
        for sequence, cl in zip(train_sequences, train_labels):
            npylm = self.npylm_classes[cl]
            npylm.add_sequence(sequence)
            if not cl in train_sequences_classes:
                train_sequences_classes[cl] = []
            train_sequences_classes[cl].append(sequence)

        # Training
        for i in range(epochs):
            for cl in classes:
                blocked_gibbs_iteration(self.npylm_classes[cl], train_sequences_classes[cl])
                apply_hyperparameters_learning(self.npylm_classes[cl], train_sequences_classes[cl], d_theta_learning, poisson_learning)

            if (i+1)%print_each_nth_iteration == 0:
                train_segments, train_perplexity, train_prediction = self.predict_segments_classes(train_data)
                dev_segments, dev_perplexity, dev_prediction = self.predict_segments_classes(dev_data)
                correct = 0.0
                for gold_y, pred_y in zip(train_labels, train_prediction):
                    if gold_y == pred_y:
                        correct += 1.0
                train_accuracy = 100.0*(correct/len(train_labels))
                correct = 0.0
                for gold_y, pred_y in zip(dev_labels, dev_prediction):
                    if gold_y == pred_y:
                        correct += 1.0
                dev_accuracy = 100.0*(correct/len(dev_labels))
                logging.info("\n{}. iteration\n\tTrain Perplexity: {}\n\tTrain Accuracy: {}\n\tDev Perplexity: {}\n\tDev Accuracy: {}".format(i+1, train_perplexity, train_accuracy, dev_perplexity, dev_accuracy)) # TODO REPLACE BY PRINT


    cpdef tuple predict_segments_classes(self, list sequences):
        """
        Call the predict_classes function. Based on its values, compute perplexity.

        Parameters
        ----------
        sequences : list of strings
            list of training sequences, each represented as a string
        Returns
        -------
        segmented_sequences : list of lists of strings
            list of sequences, each sequence is represented as list of strings
        perplexity : float
            compute perplexity of segmented sequences
        prediction : list of strings
            list of predicted classes
        """
        cdef list prediction
        cdef list segmentations
        cdef list segmentation_log_probs
        prediction, segmentations, segmentation_log_probs = self.predict_classes(sequences)
        cdef list segmentation
        cdef float log_prob
        cdef float perplexity
        cdef float prob_sum = 0.0
        for segmentation, log_prob in zip(segmentations, segmentation_log_probs):
            prob_sum += (log_prob/len(segmentation))
        perplexity = exp(-prob_sum/len(sequences))

        return segmentations, perplexity, prediction


    cpdef tuple predict_segments(self, list sequences):
        """
        Call the predict_classes function. Based on its values, compute perplexity.

        Parameters
        ----------
        sequences : list of strings
            list of training sequences, each represented as a string
        Returns
        -------
        segmented_sequences : list of lists of strings
            list of sequences, each sequence is represented as list of strings
        perplexity : float
            compute perplexity of segmented sequences
        """
        cdef list segmentations
        cdef list segmentation_log_probs
        _, segmentations, segmentation_log_probs = self.predict_classes(sequences)
        cdef list segmentation
        cdef float log_prob
        cdef float perplexity
        cdef float prob_sum = 0.0
        for segmentation, log_prob in zip(segmentations, segmentation_log_probs):
            prob_sum += (log_prob/len(segmentation))
        perplexity = exp(-prob_sum/len(sequences))

        return segmentations, perplexity


    cpdef tuple predict_classes(self, list sequences):
        """
        Predict classes for comming sequences using Bayes rule. Consider all classes and their top segmentations via viterbi algorithm.
        Choose the best one. Compute probs of rest classes for the fixed segmentation. Take argmax of class for the probability.
        Take the best segmentaiton and sum of the prob segmentation over all classes, which is a probability of sequence segmentaion.

        Parameters
        ----------
        sequences : list of strings
            list of sequences represented as strings of notes
        Returns
        -------
        prediction : list of strings
            list of predicted classes
        segmentations : list of lists of strings
            list of segmentations represented as list of string segments
        segmentation_log_probs : list of floats
            list of log probabilities of chosen segmentaitons (sum of segmentation over all classes)
        """
        cdef str cl
        cdef str cl2
        cdef dict segmented_sequences_classes = {}
        # Predict segmentations for each class
        for cl in self.npylm_classes:
            segmented_sequences_classes[cl] = viterbi_segment_sequences(self.npylm_classes[cl], sequences)
        
        # Find the best classes and its segmentations
        cdef list sequence_segmentation
        cdef list best_classes = []
        cdef str best_class
        cdef float best_prob
        cdef float mode_log_probability
        cdef float prob
        cdef int i
        cdef NPYLM npylm
        for i in range(len(sequences)):
            best_class = ""
            best_prob = -float('inf')
            for cl in self.npylm_classes:
                mode_log_probability = -float('inf')
                npylm = self.npylm_classes[cl]
                for cl2 in self.npylm_classes:
                    sequence_segmentation = segmented_sequences_classes[cl2][i]
                    mode_log_probability = np.logaddexp(mode_log_probability, npylm.get_segmentation_log_probability(sequence_segmentation))
                if mode_log_probability > best_prob:
                    best_class = cl
                    best_prob = mode_log_probability
            best_classes.append(best_class)

        # Check that there is no better class for the segmentation
        # otherwise rewrite the best class
        cdef list new_best_classes = []
        cdef str new_best_class
        for i, best_class in enumerate(best_classes):
            sequence_segmentation = segmented_sequences_classes[best_class][i]
            new_best_class = ""
            best_prob = -float('inf')
            for cl in self.npylm_classes:
                npylm = self.npylm_classes[cl]
                prob = npylm.get_segmentation_log_probability(sequence_segmentation)
                if prob > best_prob:
                    new_best_class = cl
                    best_prob = prob
            new_best_classes.append(new_best_class)

        # Compute sum of prob of best segmentation over all classes considering logarithm
        cdef list prediction = []
        cdef list segmentations = []
        cdef list segmentation_log_probs = []
        cdef float prob_sum
        for i, best_class in enumerate(new_best_classes):
            prediction.append(best_class)
            sequence_segmentation = segmented_sequences_classes[best_class][i]
            segmentations.append(sequence_segmentation)
            prob_sum = -float('inf')
            for cl in self.npylm_classes:
                npylm = self.npylm_classes[cl]
                prob_sum = np.logaddexp(prob_sum, npylm.get_segmentation_log_probability(sequence_segmentation) + np.log(1/8))
            segmentation_log_probs.append(prob_sum)

        return prediction, segmentations, segmentation_log_probs
        
