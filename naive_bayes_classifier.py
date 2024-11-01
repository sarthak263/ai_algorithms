import random
import numpy as np


def parse_data(file_name: str) -> list[list]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = line.rstrip().split(",")
        data.append(datum)
    random.shuffle(data)
    return data

def create_folds(xs: list, n: int) -> list[list[list]]:
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def evaluate(data,predictions):
    errors = sum(1 for i,obs in enumerate(data) if obs[0] != predictions[i])
    return round(errors / len(data),4)

def cross_validate(data,num_folds=5, smoothing=True):
    error_rates = []
    folds = create_folds(data, num_folds)
    folds = [np.array(fold) for fold in folds]

    for i, test_data in enumerate(folds):
        training_data = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        nbc = train(training_data,smoothing=smoothing)
        predictions = classify(nbc,test_data)
        error_rates.append(evaluate(test_data,predictions))

    return round(np.mean(error_rates),4)

def classify(nbc,observations,labeled=True):
    predictions = [predict_bayesian_classifier(observation,nbc) for observation in observations]
    return predictions

def train(training_data,smoothing=True):
    values, counts = np.unique(training_data[:, 0], return_counts=True)
    probabilities = {
        "e": {
            "class_prob": [],  # Placeholder for the prior probability of 'e'
            "attributes": {i: {} for i in range(1,len(mushroom_data[0]))}
        },
        "p": {
            "class_prob": [],  # Placeholder for the prior probability of 'p'
            "attributes": {i: {} for i in range(1,len(mushroom_data[0]))}
        }
    }

    # Store class probabilities and counts
    probabilities["e"]["class_prob"] = [counts[0] / len(training_data), counts[0]]
    probabilities["p"]["class_prob"] = [counts[1] / len(training_data), counts[1]]
    calc_probabilities(training_data,probabilities,smoothing=smoothing)
    return probabilities

def calc_probabilities(training_data, probabilities,smoothing=True):
    def calc_conditional_probabilities(data, attr_num, value, smoothing=True, ):
        values, counts = np.unique(data[:, 0], return_counts=True)
        if smoothing: counts = counts + 1

        if len(counts) == 1:
            if values[0] == "p": counts = [1, counts[0]]
            else: counts = [counts[0], 1]

        partial_prob = np.array(counts) / np.array([probabilities["e"]["class_prob"][1], probabilities["p"]["class_prob"][1]])
        probabilities["e"]["attributes"][attr_num][value] = partial_prob[0]
        probabilities["p"]["attributes"][attr_num][value] = partial_prob[1]

    for attribute in range(1,len(training_data[0])):
        values, counts = np.unique(training_data[:, attribute], return_counts=True)
        for value, count in zip(values, counts):
            subset = training_data[training_data[:, attribute] == value]
            calc_conditional_probabilities(subset,attribute,value,smoothing)

def predict_bayesian_classifier(observation,probabilities):
    prob_e = probabilities["e"]["class_prob"][0]
    prob_p = probabilities["p"]["class_prob"][0]

    for attr_num, value in enumerate(observation[1:],start=1):
        prob_e *= probabilities["e"]["attributes"][attr_num].get(value,1e-9)
        prob_p *= probabilities["p"]["attributes"][attr_num].get(value,1e-9)

    return "e" if prob_e > prob_p else "p"

mushroom_data = np.array(parse_data("agaricus-lepiota.data"))
print(f"Error Rate with smoothing {cross_validate(data=mushroom_data,num_folds=5,smoothing=True)}")
print(f"Error Rate without smoothing {cross_validate(data=mushroom_data,num_folds=5,smoothing=False)}")
