from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression


def columnwise_cosine_similarity(a, b):
    sims = []
    for idx in tqdm(range(a.shape[0])):
        a_tfidf = a[idx, :]
        b_tfidf = b[idx, :]
        sims.append(cosine_similarity(a_tfidf, b_tfidf).item())
    return sims

def logistic_regression_confidence(trained_scores, rand_scores, scores, return_classifier=False):
    """Given values of trained/actual scores and values from a random, representative distribution, and
    a new sample of score(s), return the confidence that each score is in the actual vs random distribution.
    We define the confidence of a score as the likelihood it belongs to the training distribution as
    compared to a random distribution as obtained via logistic regression.

    Args:
        trained_scores (np.array): 1D array of "actual" distribution scores as obtained possibly during training.
        rand_scores (np.array): 2D array of "shuffled" or otherwise random scores where each row is a set of
            random scores corresponding to the row in `scores`.
        scores (np.array): The recorded scores by which we want to determine confidence against the other
            distributions.
        return_classifier (boolean): If True, then return a tuple that is the array of confidence values and a list
            of classifiers for each row of scores passed in. This is useful for further evaluations using the 
            LogisticRegression classifiers in a debugging or analytic mode. Default is False.
    """
    confs = []
    classifiers = []
    X1 = trained_scores
    for idx, score in enumerate(scores):
        if np.isnan(score):
            confs.append(0.)
            continue
        X0 = rand_scores[idx]
        X = np.concatenate([X0, X1]).reshape(-1, 1)
        y = np.zeros(X.shape[0])
        y[-X1.shape[0]:] = 1
        clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=0, class_weight='balanced').fit(X, y)
        log_prob = clf.predict_proba(score.reshape(-1, 1))
        # conf = 2 * np.max(log_prob, axis=1) - 1
        conf = log_prob[:, 1]
        val = conf[0]
        if np.isnan(val):
            val = 0.
        confs.append(val)
        if return_classifier:
            classifiers.append(clf)
        
    if return_classifier:
        return np.array(confs), classifiers
    
    return np.array(confs)
