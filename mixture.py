import numpy as np
from scipy.sparse import csr_matrix
from preprocess import logsafe, normalize_exp, normalize
from collections import defaultdict


class MarkovMixture:
    def __init__(self, n_components=2, smoothing_factor=1e-6):
        self.n_components = n_components
        self.smoothing_factor = smoothing_factor
        self.transition_matrix = None
        self.emission_matrix = None
        self.states = None

    def _initialize_params(self, i, j, k):
        alpha = np.random.uniform(0, 1.5, size=(k,))
        alpha /= alpha.sum()

        beta = np.random.uniform(0, 1.5, size=(k, j))
        beta /= beta.sum(axis=-1, keepdims=True)

        gamma = np.random.uniform(0, 1.5, size=(k, j, j))
        gamma /= gamma.sum(axis=-1, keepdims=True)

        return alpha, beta, gamma

    def _unique_states(self, sequences):
        unique_states = set()
        state_index = dict()

        n_states = 0

        for seq in sequences:
            for state in seq:
                if state not in unique_states:
                    unique_states.add(state)
                    state_index[state] = n_states
                    n_states += 1
        return unique_states, state_index

    def _initialize_sufficient_statistics(self, sequences, state_index):
        n_states = len(state_index)
        n = len(sequences)

        emission_idx = []
        transition_counts = []
        row_idx = []
        col_idx = []

        for i, seq in enumerate(sequences):
            transitions = defaultdict(int)
            for j, (current_, next_) in enumerate(zip(seq, seq[1:])):
                current_ = state_index[current_]
                next_ = state_index[next_]

                if j == 0:
                    emission_idx.append(current_)

                index = current_ * n_states + next_
                transitions[index] += 1

            transition_counts += list(transitions.values())
            row_idx += [i] * len(transitions)
            col_idx += list(transitions.keys())

        emission_matrix = csr_matrix(([1] * n, (range(n), emission_idx)), shape=(n, n_states))
        transition_matrix = csr_matrix((transition_counts, (row_idx, col_idx)), shape=(n, n_states ** 2))

        return emission_matrix, transition_matrix

    def fit(self, sequences, max_iter=10000, epsilon=1e-6):
        self.states, self.state_index = self._unique_states(sequences)

        emission_matrix, transition_matrix = self._initialize_sufficient_statistics(sequences, self.state_index)

        self.n_states = len(self.states)

        i = len(sequences)
        j = self.n_states
        k = self.n_components

        alpha, beta, gamma = self._initialize_params(i, j, k)

        llhood_cache = []

        for epoch in range(max_iter):

            """
            e-step
            """
            log_alpha = logsafe(alpha).reshape((k, 1))
            log_beta = logsafe(beta)
            log_gamma = logsafe(gamma).reshape((k, j ** 2))

            z = log_alpha.T + emission_matrix.dot(log_beta.T) + transition_matrix.dot(log_gamma.T)

            llhood = np.log(np.exp(z).sum(1)).sum()
            print(llhood)

            z = normalize_exp(z, axis=1)

            if epoch >= 1:
                if np.abs(llhood - llhood_cache[-1]) <= epsilon:
                    break

            llhood_cache.append(llhood)

            """
            m-step
            """
            alpha = z.mean(axis=0)

            beta = emission_matrix.T.dot(z).T
            beta = np.maximum(beta, self.smoothing_factor)
            beta = normalize(beta, axis=1)

            gamma = transition_matrix.T.dot(z).T
            gamma = gamma.reshape((k, j, j))
            gamma = np.maximum(gamma, self.smoothing_factor)
            gamma = normalize(gamma, axis=2)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        return self

    def predict(self, sequences):
        emission_matrix, transition_matrix = self._initialize_sufficient_statistics(sequences, self.state_index)

        i = len(sequences)
        j = self.n_states
        k = self.n_components

        log_alpha = logsafe(self.alpha).reshape((k, 1))
        log_beta = logsafe(self.beta)
        log_gamma = logsafe(self.gamma).reshape((k, j ** 2))

        z = log_alpha.T + emission_matrix.dot(log_beta.T) + transition_matrix.dot(log_gamma.T)
        z = normalize_exp(z)
        return z