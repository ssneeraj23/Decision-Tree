import imp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import random 
from pprint import pprint
from helper_funs import *
from clean_split import *


def decision_tree_algorithm(df,COLUMN_HEADERS,FEATURE_TYPES,counter=0, min_samples=2, max_depth=1000):

    # data preparations
    if counter == 0:
        data = df.values
    else:
        data = df           


    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)

        return counter+1,classification


    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data,FEATURE_TYPES)
        split_column, split_value = determine_best_split(data, potential_splits,FEATURE_TYPES)
        data_below, data_above = split_data(data,FEATURE_TYPES, split_column, split_value)

        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return counter,classification

        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)

        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)

        # instantiate sub-tree
        sub_tree = {question: []}

        # find answers (recursion)
        depth_left,yes_answer = decision_tree_algorithm(data_below,COLUMN_HEADERS,FEATURE_TYPES, counter, min_samples, max_depth)
        depth_right,no_answer = decision_tree_algorithm(data_above,COLUMN_HEADERS,FEATURE_TYPES, counter, min_samples, max_depth)

        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        if(depth_right<=depth_left):
            depth_right=depth_left
        return depth_right+1,sub_tree

def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":  # feature is continuous
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer

    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)    

def calculate_accuracy(df, tree):

    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["classification_correct"] = df["classification"] == df["label"]

    accuracy = df["classification_correct"].mean()

    return accuracy*100 