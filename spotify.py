import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import methods as mt

data = pd.read_csv('SpotifyFeatures.csv') # Opens the spotifyfeatures.csv file


print(f'Songs and features per song {data.shape}')

pop_classical, n_pop, n_classical = mt.genre_split(data,'Pop','Classical')

print(f'Number of pop songs: {n_pop} \nNumber of classical songs: {n_classical}')

pop_songs, classical_songs = mt.reduce_features(pop_classical,'liveness','loudness','label')

pop_array, classical_array = mt.numpy_conversion(pop_songs, classical_songs)

pop_training, pop_test, classical_training, classical_test = mt.training_test_split(pop_array, classical_array, n_pop, n_classical, 0.8)

training_features = np.concatenate((pop_training[:,:-1], classical_training[:,:-1]), axis=0) # Combines the training data of pop and classic features 
training_labels = np.concatenate((pop_training[:,-1], classical_training[:,-1]), axis=0) # Combines the training data of the pop and classic labels
testing_features = np.concatenate((pop_test[:,:-1], classical_test[:,:-1]), axis=0) # Combines the testing data of pop and classic features 
testing_labels = np.concatenate((pop_test[:,-1], classical_test[:,-1]), axis=0) # Combines the testing data of pop and classic labels

training_features, training_labels = mt.shuffle_X_and_y(training_features,training_labels) # Shuffles the array of training data


weights, bias, losses_by_epoch = mt.sgd(training_features, training_labels, 0.8, 50) # Uses our training data to obtain weights and bias

training_pred = mt.predict(training_features,weights,bias)
test_pred = mt.predict(testing_features,weights,bias)

training_accuracy = mt.accuracy(training_labels,training_pred)
test_accuracy = mt.accuracy(testing_labels, test_pred)

print(f'Accuracy of training data: {training_accuracy}% \nAccuracy of testing data: {test_accuracy}%')

confusion_matrix = mt.confusion(testing_labels, test_pred)
print(confusion_matrix) 

plt.figure(figsize=(10, 5))
plt.plot(losses_by_epoch)
plt.title('Error per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Training error')
plt.grid(True)
plt.show()