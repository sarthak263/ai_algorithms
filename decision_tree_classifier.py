import random
from collections import namedtuple
import numpy as np
from copy import deepcopy

Node = namedtuple('node', ['attribute','label','children'])

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

def calc_entropy(data: np.ndarray) -> float:
    values, counts = np.unique(data[:, 0], return_counts=True)  # counts [num_edibles,num_poisons]
    #print(values, counts)
    p_i = counts / len(data)
    entropy = -np.sum(p_i * np.log2(p_i, where=(p_i > 0)))
    return entropy

def pick_best_attribute(data: np.ndarray, attributes: list):
    num_samples = data.shape[0]
    E_s = calc_entropy(data)
    best_gain, best_attribute = 0, -1

    for attribute in attributes:
        values, counts = np.unique(data[:, attribute], return_counts=True)
        weights = 0

        for value, count in zip(values, counts):
            subset = data[data[:, attribute] == value]
            weights += (count / num_samples) * calc_entropy(subset)

        information_gain = E_s - weights
        if information_gain > best_gain:
            best_gain = information_gain
            best_attribute = attribute

    return best_attribute

def is_homogenous(data):
    if calc_entropy(data) == 0.0:
        return True
    else:
        return False

def get_majority_label(data)-> str:
    values,counts = np.unique(data[:, 0], return_counts=True)
    majority_label = np.argmax(counts)
    return values[majority_label]

def ID3(data,attributes:list,default):
    if data is None: return Node(attribute=None,label=default,children={})
    if is_homogenous(data): return Node(attribute=None,label=data[0,0],children={})
    if attributes is None: return Node(attribute=None,label=get_majority_label(data),children={})

    best_attribute = pick_best_attribute(data,attributes)
    node: Node = Node(attribute=best_attribute,label=None,children={})
    default_label = get_majority_label(data)

    updated_attributes = deepcopy(attributes)
    updated_attributes.remove(best_attribute)
    values, counts = np.unique(data[:, best_attribute], return_counts=True)

    for value, count in zip(values,counts):
        subset = data[data[:, best_attribute] == value]
        child = ID3(subset,updated_attributes,default_label)
        node.children[value] = child

    return node

def print_tree(node,level=0):
    indent = " " * level
    if node.label is not None:
        print(f"{indent}label: {node.label}")
    else:
        print(f"{indent}attribute: {node.attribute}")
        for value,child in node.children.items():
            print(f"{indent}  Value: {value}")
            print_tree(child,level+1)

def train(training_data: np.ndarray):
    init_attributes = [i + 1 for i in range(training_data.shape[1] - 1)]
    return ID3(training_data,init_attributes,get_majority_label(training_data))

def evaluate(data,predictions):
    errors = sum(1 for i,obs in enumerate(data) if obs[0] != predictions[i])
    return round(errors / len(data),4)

def classify(tree,observations):
    classifications = []

    for observation in observations:
        node = tree
        while node.label is None:
            value = observation[node.attribute]
            if value in node.children:
                node = node.children[value]
            else:
                #unseen data
                print(get_majority_label(np.array([observation])))
                classifications.append(get_majority_label(np.array([observation])))
                break
        if node.label:
            classifications.append(node.label)
    return classifications

def cross_validate(data,num_folds=5):
    error_rates = []
    folds = create_folds(data, num_folds)
    folds = [np.array(fold) for fold in folds]

    for i, test_data in enumerate(folds):
        training_data = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        tree = train(training_data)
        predictions = classify(tree,test_data)
        error_rates.append(evaluate(test_data,predictions))

    return np.mean(error_rates)

#mushroom_data = np.array(parse_data("agaricus-lepiota.data"))
#tree = ID3(mushroom_data,attributes=[i + 1 for i in range(mushroom_data.shape[1] - 1)],default=get_majority_label(mushroom_data))
#cross_validate(parse_data('agaricus-lepiota.data'),5)

tree = Node(attribute=1, label=None, children={'x': Node(label='e', attribute=None, children={})})
obv = np.array([['e','x'], ['p','y']])
predictions = classify(tree, obv)
print(predictions)