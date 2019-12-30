import numpy as np


def simulate_hard(n=10, k=3, p=3, minlen=2, maxlen=10):
    sequences = []
    labels = []
    categories = []
    for i in range(k):
        samples = n // k

        cat = np.arange(p*i, p*(i+1))
        for _ in range(samples):
            sequences.append(list(np.random.choice(cat, size=(np.random.randint(minlen, maxlen)))))
            labels.append(i)

        categories += list(cat)

    return sequences, labels, categories

def simulate(n=10, k=3, p=3, minlen=2, maxlen=10):
    """
    Generate sequences from k components
    that obey the markov property

    :param n:
    :param K:
    :param p:
    :return:
    """

    # genearte k random initial probabilities
    beta = np.random.uniform(size=(k, p))
    beta /= beta.sum(axis=1, keepdims=True)

    # generate random K transition matrices
    gamma = np.random.uniform(size=(k, p, p))
    gamma /= gamma.sum(axis=2, keepdims=True)

    sequences = []
    labels = []
    categories = np.arange(0, p)

    # genearte sequences by drawing from the transition probabilities
    for _ in range(n):
        component = int(np.random.randint(0, k))

        current_state = np.random.choice(categories, p=beta[component])
        seq_len = np.random.randint(minlen, maxlen)

        seq = [current_state]

        for step in range(seq_len - 1):
            next_state = np.random.choice(categories, p=gamma[component, current_state])
            seq.append(next_state)
            current_state = next_state

        sequences.append(seq)
        labels.append(component)

    return sequences, labels, categories, beta, gamma
