from sklearn.metrics import classification_report


def build_classification_report(y_true, y_pred, class_names):
    return classification_report(y_true, y_pred, target_names=class_names, digits=4)

