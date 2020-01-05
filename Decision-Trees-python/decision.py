"""Implementations of decision trees and random forests"""

# Standard library imports
import bisect
import collections
import os.path
import random

# Third party imports
import numpy as np

# ______________________________________________________________________________
# Distance metrics (mathematical functions on sequences/vectors)


def euclidean_distance(x, y):
    """Euclidean (i.e. L2) metric"""
    return np.sqrt(sum((_x - _y) ** 2 for _x, _y in zip(x, y)))


def manhattan_distance(x, y):
    """L1 metric (a.k.a. New York taxi driver's metric)"""
    return sum(abs(_x - _y) for _x, _y in zip(x, y))


def hamming_distance(x, y):
    """Counts the number of coordinates on which two sequences differ"""
    return sum(_x != _y for _x, _y in zip(x, y))


def ms_error(x, y):
    """The Mean Squared Error"""
    return np.mean((_x - _y) ** 2 for _x, _y in zip(x, y))


def rms_error(x, y):
    """Square Root of the Mean Squared Error - Compare Euclidean distance (differs by scaling factor)"""
    return np.sqrt(ms_error(x, y))


def mean_error(x, y):
    """Mean absolute error - Compare Manhattan distance"""
    return np.mean(abs(_x - _y) for _x, _y in zip(x, y))


def mean_boolean_error(x, y):
    """Proportion of coordinates on which two sequences differ - Compare Hamming distance"""
    return np.mean(_x != _y for _x, _y in zip(x, y))


# ______________________________________________________________________________
# argmin and argmax


# Identity function
identity = lambda x: x


def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items


def argmin_random_tie(seq, function=identity):
    """Return an element of the input sequence with the lowest value of function(seq[i]); break ties at random.
    The input seq contains the arguments, and we output the argument which minimises function(arg) over elements in seq"""
    return min(shuffled(seq), key=function)


def argmax_random_tie(seq, function=identity):
    """Return an element of the input sequence with the highest value of function(seq[i]); break ties at random.
    The input seq contains the arguments, and we output the argument which maximises function(arg) over elements in seq"""
    return max(shuffled(seq), key=function)


# ______________________________________________________________________________
# Helper/utility functions for handling data


def remove_all(item, seq):
    """Return a copy of seq (a sequence, set, or string) with all occurrences of item removed."""
    if isinstance(seq, str):
        return seq.replace(item, '')
    elif isinstance(seq, set):
        rest = seq.copy()
        rest.remove(item)
        return rest
    else:
        return [x for x in seq if x != item]


def unique(seq):
    """Remove duplicate elements from seq. Assumes hashable elements."""
    return list(set(seq))


def count(seq):
    """Count the number of items in seq that are interpreted as true."""
    return sum(map(bool, seq))


def mode(data):
    """Return the most common data item. If there are ties, return any one of them."""
    [(item, count)] = collections.Counter(data).most_common(1)
    return item


def open_data(name, mode='r'):
    # Retrieve the root directory path
    root = os.path.dirname(__file__)
    # Join the name of the file contained in the 'data' subdirectory
    filepath = os.path.join(root, *['data', name])
    return open(filepath, mode=mode)


def num_or_str(x):
    """Used for parsing data. Convert the input string to a number if possible, or otherwise strip it."""
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()

        
def parse_csv(input, delim=','):
    r"""Parse a comma-separated values file.
    Input is a string consisting of lines, each line has comma-delimited
    fields. Convert this into a list of lists. Blank lines are skipped.
    Fields that look like numbers are converted to numbers.
    The delim defaults to ',' but tab '\t' and pipe '|' may also be useful.
    >>> parse_csv('1, 2, 3 \n 4, 5, na')
    [[1, 2, 3], [4, 5, 'na']]
    """
    lines = [line for line in input.splitlines() if line.strip()]
    return [list(map(num_or_str, line.split(delim))) for line in lines]


# ______________________________________________________________________________
# Decision Tree implementation


class DecisionFork:
    """The class for all non-leaf nodes in our decision trees.
    A fork of a decision tree holds an attribute to test, and a dict
    of branches, one for each of the attribute's values.
    """

    def __init__(self, attr, attr_name=None, default_child=None, branches=None):
        """Initialize a class instance by declaring which attribute this node tests."""
        self.attr = attr
        self.attr_name = attr_name or attr # Defaults to attr (integer index) unless a name is provided
        self.default_child = default_child # Default child is typically set to the most common attr value
        self.branches = branches or {} # Defaults to an empty dictionary unless branches provided

    def __call__(self, example):
        """Allows calling a class instance like a function. 
        Given an example, classify it using the attribute and the branches.
        Represents/returns the predicted target value of the example (recursively).
        """
        attr_val = example[self.attr]
        if attr_val in self.branches:
            return self.branches[attr_val](example)
        else:
            # return default class when attribute is unknown
            return self.default_child(example)

    def add(self, val, subtree):
        """Add a branch (subtree) for a given value of self.attr."""
        self.branches[val] = subtree

    def display(self, indent=0):
        """Pretty print the decision tree (recursively)"""
        name = self.attr_name
        print('Test', name)
        for (val, subtree) in self.branches.items():
            print(' ' * 4 * indent, name, '=', val, '==>', end=' ')
            subtree.display(indent + 1)

    def __repr__(self):
        """Formal string representation - expression for reproducing the given instance of this class"""
        return 'DecisionFork({0!r}, {1!r}, {2!r})'.format(self.attr, self.attr_name, self.branches)


class DecisionLeaf:
    """A leaf (end node with only one neighbour) of a decision tree holds just a result (prediction)."""

    def __init__(self, result):
        """Initialize a class instance by storing a result (predicted target value)"""
        self.result = result

    def __call__(self, example):
        """Returns the predicted target value of the example (to be sent back up the recursion loop)"""
        return self.result

    def display(self):
        print('RESULT =', self.result)

    def __repr__(self):
        return repr(self.result)


def DecisionTreeLearner(dataset):
    """Algorithm for training a decision tree on an input dataset and returning a trained model.
    
    The trained model is a prediction function which takes an input example and returns a predicted target value.
    """

    target, values = dataset.target, dataset.values

    def decision_tree_learning(examples, attrs, parent_examples=()):
        if len(examples) == 0:
            return plurality_value(parent_examples)
        if all_same_class(examples):
            return DecisionLeaf(examples[0][target])
        if len(attrs) == 0:
            return plurality_value(examples)
        A = choose_attribute(attrs, examples)
        tree = DecisionFork(A, dataset.attr_names[A], plurality_value(examples))
        for (v_k, exs) in split_by(A, examples):
            subtree = decision_tree_learning(exs, remove_all(A, attrs), examples)
            tree.add(v_k, subtree)
        return tree

    def plurality_value(examples):
        """
        Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality).
        """
        popular = argmax_random_tie(values[target], key=lambda v: count(target, v, examples))
        return DecisionLeaf(popular)

    def count(attr, val, examples):
        """Count the number of examples that have example[attr] = val."""
        return sum(e[attr] == val for e in examples)

    def all_same_class(examples):
        """Are all these examples in the same target class?"""
        class0 = examples[0][target]
        return all(e[target] == class0 for e in examples)

    def choose_attribute(attrs, examples):
        """Choose the attribute with the highest information gain."""
        return argmax_random_tie(attrs, key=lambda a: information_gain(a, examples))

    def information_gain(attr, examples):
        """Return the expected reduction in entropy from splitting by attr."""

        def I(examples):
            return information_content([count(target, v, examples) for v in values[target]])

        n = len(examples)
        remainder = sum((len(examples_i) / n) * I(examples_i) for (v, examples_i) in split_by(attr, examples))
        return I(examples) - remainder

    def split_by(attr, examples):
        """Return a list of (val, examples) pairs for each val of attr."""
        return [(v, [e for e in examples if e[attr] == v]) for v in values[attr]]

    return decision_tree_learning(dataset.examples, dataset.inputs)


def information_content(values):
    """Number of bits to represent the probability distribution in values."""
    probabilities = normalize(remove_all(0, values))
    return sum(-p * np.log2(p) for p in probabilities)


def normalize(dist):
    """Normalize a distribution of counts so that their sum is 1.0, thus representing proportions/probabilities.
    This is achieved by multiplying each number by a constant such that the sum is 1.0 (i.e. dividing by the original total)"""
    if isinstance(dist, dict): # Implementation for dictionaries
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key] / total # Assign the normalized values
            assert 0 <= dist[key] <= 1  # Check that our probabilities are between 0 and 1
        return dist
    else: # Implementation for lists, tuples
        total = sum(dist)
        return [(n / total) for n in dist]


# ______________________________________________________________________________
# Random Forest implementation


def probability(p):
    """Return true with probability p. 
    Used to simulate sampling a discrete random variable distributed as Bernoulli(p)."""
    return p > random.uniform(0.0, 1.0)


def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights.
    Used to simulate sampling from a discrete random variable with categorical distribution according to the weights.
    Check the docs at https://docs.python.org/3/library/bisect.html to understand more about the implementation"""
    totals = []
    # Fill totals with the cumulative sums of the weights
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]


def weighted_sample_with_replacement(n, seq, weights):
    """Pick n samples from seq at random, with replacement, 
    with the probability of sampling each element in proportion to its corresponding weight.
    
    Used to simulate n independent samples from a discrete random variable with categorical distribution according to the weights.
    """
    sample = weighted_sampler(seq, weights)
    return [sample() for _ in range(n)]


def RandomForest(dataset, n=5):
    """An ensemble of n Decision Trees trained using data bagging and feature bagging.
    
    We train each decision tree using a subset of the total examples sampled randomly with replacement, so repetitions allowed,
    and allowing only a subset of the features to be used to create the tree (subset of allowable features selected randomly).
    
    The trained Random Forest model is an ensemble model as it takes an input example and calculates the prediction for each
    decision tree within the ensemble, then returns the most common predicted target value as the final output.
    """

    def data_bagging(dataset, m=0):
        """Sample m examples, with replacement, from the dataset.
        
        Examples have equal probability of being sampled, and the number of samples defaults to the size, n, of the dataset.
        """
        n = len(dataset.examples)
        return weighted_sample_with_replacement(m or n, dataset.examples, [1] * n)

    def feature_bagging(dataset, p=0.7):
        """Feature bagging with probability p to retain an attribute
        
        Returns a subset of dataset.inputs where each attribute is included in the subset with probability p (default p=0.7).
        If an empty subset is selected by chance, we instead use all of the inputs.
        """
        inputs = [i for i in dataset.inputs if probability(p)]
        return inputs or dataset.inputs

    # Construct the randomly selected forest of individual decision trees
    predictors = [DecisionTreeLearner(DataSet(examples=data_bagging(dataset), attrs=dataset.attrs,
                                              attr_names=dataset.attr_names, target=dataset.target,
                                              inputs=feature_bagging(dataset))) for _ in range(n)]   
    
    def predict(example):
        """Given an input example, the predicted target value of the random forest model is the most common
        predicted target value amongst its component decision trees.
        We print the individual predictions and return their mode."""
        print([predictor(example) for predictor in predictors])
        return mode(predictor(example) for predictor in predictors)

    return predict


# ______________________________________________________________________________
# Dataset class definition


class DataSet:
    """
    A data set for a machine learning problem. It has the following fields:
    d.examples   A list of examples. Each one is a list of attribute values.
    d.attrs      A list of integers to index into an example, so example[attr]
                 gives a value. Normally the same as range(len(d.examples[0])).
    d.attr_names Optional list of mnemonic names for corresponding attrs.
    d.target     The attribute that a learning algorithm will try to predict.
                 By default the final attribute.
    d.inputs     The list of attrs without the target.
    d.values     A list of lists: each sublist is the set of possible
                 values for the corresponding attribute. If initially None,
                 it is computed from the known examples by self.set_problem.
                 If not None, an erroneous value raises ValueError.
    d.distance   A function from a pair of examples to a non-negative number.
                 Should be symmetric, etc. Defaults to mean_boolean_error
                 since that can handle any field types.
    d.name       Name of the data set (for output display only).
    d.source     URL or other source where the data came from.
    d.exclude    A list of attribute indexes to exclude from d.inputs. Elements
                 of this list can either be integers (attrs) or attr_names.
    Normally, you call the constructor and you're done; then you just
    access fields like d.examples and d.target and d.inputs.
    """

    def __init__(self, examples=None, attrs=None, attr_names=None, target=-1, inputs=None,
                 values=None, distance=mean_boolean_error, name='', source='', exclude=()):
        """
        Accepts any of DataSet's fields. Examples can also be a
        string or file from which to parse examples using parse_csv.
        Optional parameter: exclude, as documented in .set_problem().
        >>> DataSet(examples='1, 2, 3')
        <DataSet(): 1 examples, 3 attributes>
        """
        self.name = name
        self.source = source
        self.values = values
        self.distance = distance
        self.got_values_flag = bool(values)

        # initialize .examples from string or list or data directory
        if isinstance(examples, str):
            self.examples = parse_csv(examples)
        elif examples is None:
            self.examples = parse_csv(open_data(name + '.csv').read())
        else:
            self.examples = examples

        # attrs are the indices of examples, unless otherwise stated.
        if self.examples is not None and attrs is None:
            attrs = list(range(len(self.examples[0])))

        self.attrs = attrs

        # initialize .attr_names from string, list, or by default
        if isinstance(attr_names, str):
            self.attr_names = attr_names.split()
        else:
            self.attr_names = attr_names or attrs
        self.set_problem(target, inputs=inputs, exclude=exclude)

    def set_problem(self, target, inputs=None, exclude=()):
        """
        Set (or change) the target and/or inputs.
        This way, one DataSet can be used multiple ways. inputs, if specified,
        is a list of attributes, or specify exclude as a list of attributes
        to not use in inputs. Attributes can be -n .. n, or an attr_name.
        Also computes the list of possible values, if that wasn't done yet.
        """
        self.target = self.attr_num(target)
        exclude = list(map(self.attr_num, exclude))
        if inputs:
            self.inputs = remove_all(self.target, inputs)
        else:
            self.inputs = [a for a in self.attrs if a != self.target and a not in exclude]
        if not self.values:
            self.update_values()
        self.check_me()

    def check_me(self):
        """Check that my fields make sense."""
        assert len(self.attr_names) == len(self.attrs)
        assert self.target in self.attrs
        assert self.target not in self.inputs
        assert set(self.inputs).issubset(set(self.attrs))
        if self.got_values_flag:
            # only check if values are provided while initializing DataSet
            list(map(self.check_example, self.examples))

    def add_example(self, example):
        """Add an example to the list of examples, checking it first."""
        self.check_example(example)
        self.examples.append(example)

    def check_example(self, example):
        """Raise ValueError if example has any invalid values."""
        if self.values:
            for a in self.attrs:
                if example[a] not in self.values[a]:
                    raise ValueError('Bad value {} for attribute {} in {}'
                                     .format(example[a], self.attr_names[a], example))

    def attr_num(self, attr):
        """Returns the number used for attr, which can be a name, or -n .. n-1."""
        if isinstance(attr, str):
            return self.attr_names.index(attr)
        elif attr < 0:
            return len(self.attrs) + attr
        else:
            return attr

    def update_values(self):
        self.values = list(map(unique, zip(*self.examples)))

    def sanitize(self, example):
        """Return a copy of example, with non-input attributes replaced by None."""
        return [attr_i if i in self.inputs else None for i, attr_i in enumerate(example)]

    def classes_to_numbers(self, classes=None):
        """Converts class names to numbers."""
        if not classes:
            # if classes were not given, extract them from values
            classes = sorted(self.values[self.target])
        for item in self.examples:
            item[self.target] = classes.index(item[self.target])

    def remove_examples(self, value=''):
        """Remove examples that contain given value."""
        self.examples = [x for x in self.examples if value not in x]
        self.update_values()

    def split_values_by_classes(self):
        """Split values into buckets according to their class."""
        buckets = defaultdict(lambda: [])
        target_names = self.values[self.target]

        for v in self.examples:
            item = [a for a in v if a not in target_names]  # remove target from item
            buckets[v[self.target]].append(item)  # add item to bucket of its class

        return buckets

    def find_means_and_deviations(self):
        """
        Finds the means and standard deviations of self.dataset.
        means     : a dictionary for each class/target. Holds a list of the means
                    of the features for the class.
        deviations: a dictionary for each class/target. Holds a list of the sample
                    standard deviations of the features for the class.
        """
        target_names = self.values[self.target]
        feature_numbers = len(self.inputs)

        item_buckets = self.split_values_by_classes()

        means = defaultdict(lambda: [0] * feature_numbers)
        deviations = defaultdict(lambda: [0] * feature_numbers)

        for t in target_names:
            # find all the item feature values for item in class t
            features = [[] for _ in range(feature_numbers)]
            for item in item_buckets[t]:
                for i in range(feature_numbers):
                    features[i].append(item[i])

            # calculate means and deviations fo the class
            for i in range(feature_numbers):
                means[t][i] = mean(features[i])
                deviations[t][i] = stdev(features[i])

        return means, deviations

    def __repr__(self):
        return '<DataSet({}): {:d} examples, {:d} attributes>'.format(self.name, len(self.examples), len(self.attrs))
        

# ______________________________________________________________________________
# Creating an example dataset

def RestaurantDataSet(examples=None):
    """
    [Figure 18.3]
    Build a DataSet of Restaurant waiting examples.
    """
    return DataSet(name='restaurant', target='Wait', examples=examples,
                   attr_names='Alternate Bar Fri/Sat Hungry Patrons Price Raining Reservation Type WaitEstimate Wait')


restaurant = RestaurantDataSet()


def T(attr_name, branches):
    branches = {value: (child if isinstance(child, DecisionFork) else DecisionLeaf(child))
                for value, child in branches.items()}
    return DecisionFork(restaurant.attr_num(attr_name), attr_name, print, branches)


""" 
[Figure 18.2]
A decision tree for deciding whether to wait for a table at a hotel.
"""

waiting_decision_tree = T('Patrons',
                          {'None': 'No', 'Some': 'Yes',
                           'Full': T('WaitEstimate',
                                     {'>60': 'No', '0-10': 'Yes',
                                      '30-60': T('Alternate',
                                                 {'No': T('Reservation',
                                                          {'Yes': 'Yes',
                                                           'No': T('Bar', {'No': 'No',
                                                                           'Yes': 'Yes'})}),
                                                  'Yes': T('Fri/Sat', {'No': 'No', 'Yes': 'Yes'})}),
                                      '10-30': T('Hungry',
                                                 {'No': 'Yes',
                                                  'Yes': T('Alternate',
                                                           {'No': 'Yes',
                                                            'Yes': T('Raining',
                                                                     {'No': 'No',
                                                                      'Yes': 'Yes'})})})})})


def SyntheticRestaurant(n=20):
    """Generate a DataSet with n examples."""

    def gen():
        example = list(map(random.choice, restaurant.values))
        example[restaurant.target] = waiting_decision_tree(example)
        return example

    return RestaurantDataSet([gen() for _ in range(n)])

