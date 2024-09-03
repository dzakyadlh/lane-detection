import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Provided values
TP = round(2978/3028, 3)
FP = round(40/3028, 3)
FN = round(10/3028, 3)

# Placeholder for True Negatives (TN)
TN = 0  # Set to zero or some other value if necessary

# Create a confusion matrix array
cm = np.array([[TP, FP], 
               [FN, TN]])

# Plot the confusion matrix using scikit-learn
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Paddy', 'Background'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
