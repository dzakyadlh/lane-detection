import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Provided values
TP = 1465
FP = 286
FN = 105

# Placeholder for True Negatives (TN)
TN = 0  # Set to zero or some other value if necessary

# Create a confusion matrix array
cm = np.array([[TP, FP], 
               [FN, TN]])

# Plot the confusion matrix using scikit-learn
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Positive', 'Negative'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
