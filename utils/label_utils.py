from sklearn.metrics import confusion_matrix









def get_classification_metrics(true_labels, predicted_labels):
    """
    Calculate False Positives (FP), False Negatives (FN), and Correctly Classified (CC)
    based on the true labels and predicted labels.
    
    Args:
    - true_labels (list or np.array): Ground truth labels.
    - predicted_labels (list or np.array): Model's predicted labels.
    
    Returns:
    - tuple: (False Positives, False Negatives, Correctly Classified)
    """
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    
    # Correctly classified are those that are either True Positives or True Negatives
    cc = tn + tp
    
    return fp, fn, cc