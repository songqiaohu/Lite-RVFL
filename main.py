import numpy as np
from skmultiflow.data import FileStream, HyperplaneGenerator, WaveformGenerator, SEAGenerator
from clf_Lite_RVFL import Lite_RVFL
import matplotlib.pyplot as plt
import time
import warnings
import random

# Suppress warnings
warnings.filterwarnings("ignore")

# Set the seed for reproducibility
# np.random.seed(70) #70
# random.seed(70)  #70

# Initialize the data stream
stream = FileStream('DSMS.csv')
x_train, y_train = stream.next_sample(200)

# Initialize the classifier
clf = Lite_RVFL(Ne=10,
                 N2=10,
                 enhence_function='sigmoid',
                 reg=0.1,
                 theta = 1.003)  #1.0018
clf.fit(x_train, y_train)

# Variables to keep track of accuracy
count = 0
correct = 0
accuracy = []  # For cumulative accuracy
current_accuracy = []  # For current accuracy (every 100 samples)
window_size = 100  # Number of samples for current accuracy

# Store predictions and labels for the last 100 samples
predictions = []
labels = []

# Start timing
t1 = time.time()

# Loop through the data stream
while count < 30000-200:
    x, y = stream.next_sample()
    count += 1
    print(count)

    # Make predictions
    y_pred = clf.predict(x)

    # Update the predictions and labels
    predictions.extend(y_pred)
    labels.extend(y)

    # Update the classifier incrementally
    clf.partial_fit(x, y)

    # Cumulative accuracy: All samples processed so far
    correct += np.sum(y_pred == y)
    accuracy.append(correct / count)
    # print(clf.W)

    # Calculate current accuracy every 100 samples
    if count % window_size == 0:
        correct_in_window = sum(np.array(predictions[-window_size:]) == np.array(labels[-window_size:]))
        current_accuracy.append(correct_in_window / window_size)

    # if count % 7000 == 0:
    #     x_retrain, y_retrain = stream.next_sample(500)
    #     clf = CDA_RVFL()
    #     clf.fit(x_retrain, y_retrain)

# End timing
t2 = time.time()
print(f'Time = {t2 - t1}s')

# Print final accuracy
print(f'Final Cumulative Accuracy = {correct / count}')

# Plot both cumulative and current accuracy
plt.figure(figsize=(10, 6))

# Plot cumulative accuracy
plt.plot(range(1, count + 1), accuracy, label='Cumulative Accuracy', color='blue')

# Plot current accuracy every 100 samples
plt.plot(range(window_size, count + 1, window_size), current_accuracy, label='Current Accuracy (last 100 samples)', color='red')

# Labels and title
plt.xlabel('Samples Processed')
plt.ylabel('Accuracy')
plt.title('Cumulative vs Current Accuracy')
plt.legend()

# Show plot
plt.show()
