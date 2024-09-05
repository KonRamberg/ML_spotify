import numpy as np
import pandas as pd

def genre_split(data, genre1, genre2): # Splits the data into two genres we want to use
    new_list = data[data['genre'].isin([genre1, genre2])].copy()
    new_list['label'] = new_list['genre'].apply(lambda x: 1 if x == genre1 else 0) # Adds labels as features to  the genres we picked

    n_genre1 = (new_list['label']==1).sum() # Amount of songs in the first genre
    n_genre2 = (new_list['label']==0).sum() # Amount of songs in the second genre
    
    return new_list, n_genre1, n_genre2 

def reduce_features(data,feature1,feature2,feature3): # Reduces the features to those we want to use from the dataset
    reduced_dataset = data[[feature1,feature2,feature3]]
    
    genre1_songs= reduced_dataset[reduced_dataset['label'].isin([1])].copy() # Creates two datasets out of the data we have left
    genre2_songs = reduced_dataset[reduced_dataset['label'].isin([0])].copy()
    
    genre1_songs = genre1_songs.sample(frac=1) # Shuffles the datasets
    genre2_songs = genre2_songs.sample(frac=1)

    return genre1_songs, genre2_songs

def numpy_conversion(genre1_songs,genre2_songs): # Converts two datasets to numpy arrays
    genre1_array = genre1_songs.to_numpy()
    genre2_array = genre2_songs.to_numpy()

    return genre1_array, genre2_array

def training_test_split(genre1_array, genre2_array, n_genre1, n_genre2, split): # Splits the two datasets into the ratio split%/(100-split)% for training and testing
    genre1_index = int(n_genre1 * split) # Index calculation
    genre2_index = int(n_genre2 * split)

    genre1_training = genre1_array[:genre1_index]
    genre1_test = genre1_array[genre1_index:]
    
    genre2_training = genre2_array[:genre2_index]
    genre2_test = genre2_array[genre2_index:]
    
    return genre1_training, genre1_test, genre2_training, genre2_test

def sigmoid(z): # Basic sigmoid calculation which maps any number to a value between 0 and 1
    return 1/(1+np.exp(-z))

def shuffle_X_and_y(X_array, y_array): # Shuffles two arrays of the same size, in the same order
    shuffle = np.random.permutation(len(y_array))
    X_array = X_array[shuffle]
    y_array = y_array[shuffle]
    return X_array, y_array

def log_loss(y_true, y_pred): # Binary cross-entropy loss calculation measures how well the prediction matches the real values
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def predict(X_array, weights, bias): # Uses the numbers obtained during training and maps each sample to a label
    logistic_model = np.dot(X_array, weights)+bias # Maps the weights to the features, adds the bias to get a prediction function
    y_pred = sigmoid(logistic_model)
    y_pred_class = [0 if i < 0.5 else 1 for i in y_pred]
    return y_pred_class

def accuracy(y_true, y_pred): # Measures the accuracy by comparing the prediction to the real values
    true_pred_compare = y_true == y_pred
    correct_count = np.sum(true_pred_compare)
    accuracy = correct_count / len(y_true)
    return round(accuracy*100,2)

def confusion(y_test, y_pred): # Element wise comparision of the prediction and the real values, 
    tp = 0  # True positive
    tn = 0  # True negative
    fp = 0  # False positive
    fn = 0  # False negative
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i] == 1: 
            tp += 1
        elif y_test[i] == y_pred[i] == 0:
            tn += 1
        elif y_test[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_test[i] == 1 and y_pred[i] == 0:
            fn += 1
        confusion_matrix = [[tn,fp], [fn,tp]] # Put the values in a matrix
    return confusion_matrix

def sgd(X_array,y_array,learning_rate,epochs): # Stochastic gradient descent to optimize weights
    samples, features = X_array.shape
    weights = np.zeros(features) # Initializes weights and bias to 0
    bias = 0
    loss_by_epoch = []
    for epoch in range(epochs):
        X_shuffled, y_shuffled = shuffle_X_and_y(X_array,y_array)
        for i in range(0,samples):
            X_i = X_shuffled[i] # Current sample features
            y_i = y_shuffled[i] # Current sample label
            
            y_pred = sigmoid(np.dot(X_i, weights) + bias) # Use the current values for our prediction function
            
            error = y_pred - y_i # Error function
            
            w_gradient = X_i * error / samples # Weight gradient
            b_gradient = error / samples # Bias gradient
            
            weights -= learning_rate * w_gradient # Weight is updated
            bias -= learning_rate * b_gradient # Weight is updated
            
        y_pred_all = sigmoid(np.dot(X_shuffled, weights) + bias)
        current_loss = log_loss(y_shuffled, y_pred_all) # Binary cross-entropy loss calculation
        print(f'Loss after epoch {epoch}: {current_loss}') # To print out the  loss at the current epoch
        loss_by_epoch.append(current_loss) # Adds the loss to the list we use to create the plot
    
    return weights, bias, loss_by_epoch