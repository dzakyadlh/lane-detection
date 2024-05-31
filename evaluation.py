from sklearn import metrics
import matplotlib.pyplot as plt

def compute_confusion_matrix(true_labels, predictions, class_names):
    y_true = []
    y_pred = []

    for true, pred in zip(true_labels, predictions):
        y_true.extend(true)
        y_pred.extend(pred)

    cm = metrics.confusion_matrix(y_true, y_pred, labels=class_names)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    cm_display.plot()
    plt.savefig('confusion_matrix.png')

def read_true_labels(label_file):
    true_labels = []
    with open(label_file, 'r') as f:
        for line in f:
            true_labels.append(line.strip().split())
    return true_labels