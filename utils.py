from collections import defaultdict

import numpy as np
import torch



def is_same_model(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def binary_accuracy(preds, y):
    """
    from text_classifier
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def multi_class_accuracy(preds, y):
    """
    from text_classifier
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def int2ordinal(num):
    """
    Convert a natural number to an ordinal number.
    """
    assert isinstance(num, int) and num >= 0
    ordinal_dict = defaultdict(lambda: "th")
    ordinal_dict.update({
        1: "st", 2: "nd", 3: "rd",
    })
    q, mod = divmod(num, 10)
    if q % 10 == 1:  # 11th
        suffix = "th"
    else:
        suffix = ordinal_dict[mod]
    return f'{num}{suffix}'


def remove_suffix(string, suffix):
        # suffix='' should not call self[:-0]
        if suffix and string.endswith(suffix):
            return string[:-len(suffix)]
        return string[:]


def get_smallest_arg(a, n):
    """
    from glue_select/elbow.py
    return indices of smallest `n` elements in `a`
    """
    if len(a) < n:
        raise ValueError(f'n(={n}) out of bounds ({len(a)})')
    elif len(a) == n:
        return np.arange(n)
    indices = np.argpartition(a, n)
    return indices[:n]


def get_elbow(values, threshold=0.25):
    """from auto-annotator/pipeline/active_labeler.py"""
    key = None
    for key, val in enumerate(values):
        if (val - values[-1]) < threshold * (values[0] - values[-1]):
            break
    return key
