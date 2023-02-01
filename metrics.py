from scipy.stats import entropy


def mean_substraction_metric(probs_dict, group):

    # 1/ Get the probability of the group of interest
    group_prob = probs_dict[group]

    # 2/ Compute the mean probability of all other groups
    mean_prob = sum([p for k, p in probs_dict.items() if k != group]) / (len(probs_dict) - 1)

    # 3/ Substract
    return group_prob - mean_prob




def entropy_metric(probs_dict, group):

    # 1/ Compute the entorpy of the true probability distribution
    dist_entropy = entropy(list(probs_dict.values()))

    # 2/ Compute the entropy of a uniform distribution with the same number of logits as in the true distribution
    uniform_entropy = entropy([1/len(probs_dict)] * len(probs_dict))

    # 3/ Decide wether it's a stereotype or an anti-stereotype
    stereotype_sign = 1 if probs_dict[group] > 1/len(probs_dict) else -1

    # 4/ Substract
    return stereotype_sign * (uniform_entropy - dist_entropy)

