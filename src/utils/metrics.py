from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import editdistance
import numpy as np

def calculate_precision(ground_truth, prediction):
    """
    Calculate the precision for an OCR model using Levenshtein distance.

    :param ground_truth: The actual text that should be identified.
    :param prediction: The text identified by the OCR model.
    :return: Precision value.
    """
    # Ensure inputs are strings
    ground_truth = str(ground_truth)
    prediction = str(prediction)
    
    # Calculate the total retrieved instances (all characters in prediction)
    total_retrieved = len(prediction)
    
    # Calculate the Levenshtein distance (number of edits) between the ground truth and the prediction
    levenshtein_dist = editdistance.eval(ground_truth, prediction)
    
    # Calculate true positives
    true_positives = len(ground_truth) - levenshtein_dist
    
    # Calculate false positives
    false_positives = total_retrieved - true_positives
    
    # Calculate precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    
    return precision


def calculate_recall(ground_truth, prediction):
    """
    Calculate the recall for an OCR model using Levenshtein distance.

    :param ground_truth: The actual text that should be identified.
    :param prediction: The text identified by the OCR model.
    :return: Recall value.
    """
    # Ensure inputs are strings
    ground_truth = str(ground_truth)
    prediction = str(prediction)
    
    # Calculate the total relevant instances (all characters in ground_truth)
    total_relevant = len(ground_truth)
    
    # Calculate the Levenshtein distance (number of edits) between the ground truth and the prediction
    levenshtein_dist = editdistance.eval(ground_truth, prediction)
    
    # Calculate true positives
    true_positives = total_relevant - levenshtein_dist
    
    # Calculate false negatives
    false_negatives = total_relevant - true_positives
    
    # Calculate recall
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    
    return recall


def get_confidences_and_labels(ground_truth, predictions, confidences):
    """
    Create arrays of true labels and confidence scores for each character in the predictions.

    :param ground_truth: The actual text that should be identified.
    :param predictions: The text identified by the OCR model.
    :param confidences: The confidence scores for each character in the predictions.
    :return: Arrays of true labels and confidence scores.
    """
    true_labels = []
    conf_scores = []
    
    min_length = min(len(ground_truth), len(predictions))
    for i in range(min_length):
        true_labels.append(1 if ground_truth[i] == predictions[i] else 0)
        conf_scores.append(confidences[i])
        
    return np.array(true_labels), np.array(conf_scores)


def accumulate_roc_data(batch_ground_truth, batch_predictions, batch_confidences):
    """
    Accumulate true labels and confidence scores from a batch of data.

    :param batch_ground_truth: List of ground truth texts for the batch.
    :param batch_predictions: List of predicted texts for the batch.
    :param batch_confidences: List of confidence scores for each text in the batch.
    :return: Arrays of accumulated true labels and confidence scores.
    """
    all_true_labels = []
    all_conf_scores = []
    
    for gt, pred, conf in zip(batch_ground_truth, batch_predictions, batch_confidences):
        true_labels, conf_scores = get_confidences_and_labels(gt, pred, conf)
        all_true_labels.extend(true_labels)
        all_conf_scores.extend(conf_scores)
        
    return np.array(all_true_labels), np.array(all_conf_scores)