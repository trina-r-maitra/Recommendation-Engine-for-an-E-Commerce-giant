import numpy as np
import time




class LogisticMF():

    def __init__(self, counts, num_factors, reg_param, gamma, iterations):
        self.counts = counts
        self.num_gsts = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.iterations = iterations
        self.reg_param = reg_param
        self.gamma = gamma

    def train_model(self):

        self.ones = np.ones((self.num_gsts, self.num_items))
        self.gst_vectors = np.random.normal(size=(self.num_gsts,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))
        self.gst_biases = np.random.normal(size=(self.num_gsts, 1))
        self.item_biases = np.random.normal(size=(self.num_items, 1))

        gst_vec_deriv_sum = np.zeros((self.num_gsts, self.num_factors))
        item_vec_deriv_sum = np.zeros((self.num_items, self.num_factors))
        gst_bias_deriv_sum = np.zeros((self.num_gsts, 1))
        item_bias_deriv_sum = np.zeros((self.num_items, 1))
        for i in range(self.iterations):
            t0 = time.time()

            gst_vec_deriv, gst_bias_deriv = self.deriv(True)
            gst_vec_deriv_sum += np.square(gst_vec_deriv)
            gst_bias_deriv_sum += np.square(gst_bias_deriv)
            vec_step_size = self.gamma / np.sqrt(gst_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(gst_bias_deriv_sum)
            self.gst_vectors += vec_step_size * gst_vec_deriv
            self.gst_biases += bias_step_size * gst_bias_deriv


            item_vec_deriv, item_bias_deriv = self.deriv(False)
            item_vec_deriv_sum += np.square(item_vec_deriv)
            item_bias_deriv_sum += np.square(item_bias_deriv)
            vec_step_size = self.gamma / np.sqrt(item_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(item_bias_deriv_sum)
            self.item_vectors += vec_step_size * item_vec_deriv
            self.item_biases += bias_step_size * item_bias_deriv
            t1 = time.time()


    def deriv(self, gst):
        if gst:
            vec_deriv = np.dot(self.counts, self.item_vectors)
            bias_deriv = np.expand_dims(np.sum(self.counts, axis=1), 1)

        else:
            vec_deriv = np.dot(self.counts.T, self.gst_vectors)
            bias_deriv = np.expand_dims(np.sum(self.counts, axis=0), 1)
        A = np.dot(self.gst_vectors, self.item_vectors.T)
        A += self.gst_biases
        A += self.item_biases.T
        A = np.exp(A)
        A /= (A + self.ones)
        A = (self.counts + self.ones) * A

        if gst:
            vec_deriv -= np.dot(A, self.item_vectors)
            bias_deriv -= np.expand_dims(np.sum(A, axis=1), 1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.gst_vectors
        else:
            vec_deriv -= np.dot(A.T, self.gst_vectors)
            bias_deriv -= np.expand_dims(np.sum(A, axis=0), 1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.item_vectors
        return (vec_deriv, bias_deriv)


        
    def print_probabilities(self):
        probs = np.zeros((self.num_gsts, self.num_items))
        expTerm = np.exp(np.add(np.add(np.dot(self.gst_vectors, self.item_vectors.T),self.gst_biases),self.item_biases.T))
        probs = expTerm/(1+expTerm)
        return probs
    
    
