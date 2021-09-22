from scipy import sparse 
from typing import List, Set, Dict, Tuple, Optional
import numpy as np

def create_one_hot(labels, label_dict: dict):
    """
    
    Args:
        labels:        array of labels, e.g. NumPy array or Pandas Series
        label_dict:    dict of label indices
    Return:
        one_hot_numpy: sparse CSR 2d array of one-hot vectors
    """
    one_hot_numpy = sparse.dok_matrix((len(labels), len(label_dict)), dtype=np.int8)
    for i, label in enumerate(labels):
        one_hot_numpy[i, label_dict[label]] = 1
    return sparse.csr_matrix(one_hot_numpy) 

def undo_one_hot(pred, label_list: list) -> List[List[str]]:
    """
    
    Args: 
        pred:       NumPy array of one-hot predicted classes
        label_list: a list of the label strings
    Return:
        label_pred: a list of predicted labels
    """
    label_pred = [label_list[np.argmax(row)] for row in pred]
    return label_pred
    # this could probably be done awesomely fast as NumPy vectorised but it works


def word_index(los: List[List[str]], vocab_dict: Dict[str, int], unknown: int, reverse: bool=False) -> List[List[int]]:
    """
    Replaces words with integers from a vocabulary dictionary or else with the integer for unknown
    
    Args:
        los:     list of lists of split sentences
        pad_to:  how big to make the padded list
        unknown: the integer to put in for unknown tokens (either because they were pruned or not seen in training set)
        reverse: reverse the order of tokens in the sub-list 
    Returns: 
        new_los: list of lists of split sentences where each token is replaced by an integer
        
    Examples:
    >>> print(word_index([['one', 'two', 'three'], ['one', 'two']], {'one': 1, 'two': 2, 'three': 3}, unknown=4))
    [[1, 2, 3], [1, 2]]
    >>> print(word_index([['one', 'two', 'three'], ['one', 'two']], {'one': 1, 'two': 2, 'three': 3}, unknown=4, reverse=True))
    [[3, 2, 1], [2, 1]]
    """
    new_los = []
    if reverse:
        for sentence in los:
            new_los.append([vocab_dict[word] if word in vocab_dict else unknown for word in sentence][::-1])        
    else:
        for sentence in los:
            new_los.append([vocab_dict[word] if word in vocab_dict else unknown for word in sentence])
    return new_los

