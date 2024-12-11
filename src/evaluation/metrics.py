""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                                                                                                    │
  │ metrics.py                                                                                                         │
  │                                                                                                                    │
  │ This module provides various metrics for evaluating the performance of a model.                                    │
  │ The metrics include word accuracy, character accuracy, and Levenshtein distance.                                   │
  │                                                                                                                    │
  │ Metrics:                                                                                                           │
  │ - Word Accuracy: Measures the accuracy at the word level.                                                          │
  │ - Character Accuracy: Measures the accuracy at the character level.                                                │
  │ - Levenshtein Distance: Measures the edit distance between predicted and target sequences.                         │
  │                                                                                                                    │
  │ Formulas:                                                                                                          │
  │ - Word Accuracy: (Number of correct words) / (Total number of words)                                               │
  │ - Character Accuracy: (Number of correct characters) / (Total number of characters)                                │
  │ - Levenshtein Distance: Minimum number of single-character edits (insertions, deletions, or substitutions)         │
  │ required to change one word into the                                                                               │
  │ other.                                                                                                             │
  │                                                                                                                    │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """


import numpy as np

def word_accuracy(preds, targets):
    """
    Calculate the word accuracy.

    Word Accuracy = (Number of correct words) / (Total number of words)

    This metric measures the proportion of words that are predicted correctly in their entirety.
    It is useful for evaluating OCR systems where the goal is to correctly recognize whole words.

    Args:
        preds (list of str): List of predicted words.
        targets (list of str): List of target words.

    Returns:
        float: Word accuracy.
    """
    correct_words = 0
    total_words = len(targets)

    for pred, target in zip(preds, targets):
        if pred == target:
            correct_words += 1

    return correct_words / total_words if total_words > 0 else 0

def character_accuracy(preds, targets):
    """
    Calculate the character accuracy.

    Character Accuracy = (Number of correct characters) / (Total number of characters)

    This metric measures the proportion of characters that are predicted correctly.
    It is useful for evaluating OCR systems where the goal is to minimize character-level errors.

    Args:
        preds (list of str): List of predicted words.
        targets (list of str): List of target words.

    Returns:
        float: Character accuracy.
    """
    correct_chars = 0
    total_chars = 0

    for pred, target in zip(preds, targets):
        correct_chars += sum(p == t for p, t in zip(pred, target))
        total_chars += len(target)

    return correct_chars / total_chars if total_chars > 0 else 0

def levenshtein_distance(pred, target):
    """
    Calculate the Levenshtein distance between two words.

    Levenshtein Distance: Minimum number of single-character edits (insertions, deletions, or substitutions) required to change one word into the other.

    This metric measures how dissimilar two strings (e.g., the predicted and target words) are.
    It is useful for evaluating OCR systems as it provides a measure of the edit effort needed to correct the predictions.

    Args:
        pred (str): Predicted word.
        target (str): Target word.

    Returns:
        int: Levenshtein distance.
    """
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif pred[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]

def average_levenshtein_distance(preds, targets):
    """
    Calculate the average Levenshtein distance for a list of predicted and target words.

    This metric provides an average measure of the edit effort needed to correct the predictions across multiple words.
    It is useful for evaluating OCR systems as it gives an overall sense of how close the predictions are to the targets.

    Args:
        preds (list of str): List of predicted words.
        targets (list of str): List of target words.

    Returns:
        float: Average Levenshtein distance.
    """
    total_distance = 0
    total_words = len(targets)

    for pred, target in zip(preds, targets):
        total_distance += levenshtein_distance(pred, target)

    return total_distance / total_words if total_words > 0 else 0



def character_confusion_matrix(preds, targets, label2char):
    """
    Calculate the character confusion matrix.

    This function computes a confusion matrix for character-level predictions.

    Args:
        preds (list of str): List of predicted words.
        targets (list of str): List of target words.
        label2char (dict): Dictionary mapping label indices to characters.

    Returns:
        np.ndarray: Confusion matrix.
    """
    all_chars = list(label2char.values())
    char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
    matrix_size = len(all_chars)
    confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    for pred, target in zip(preds, targets):
        for p, t in zip(pred, target):
            if p in char_to_idx and t in char_to_idx:
                confusion_matrix[char_to_idx[t], char_to_idx[p]] += 1

    return confusion_matrix