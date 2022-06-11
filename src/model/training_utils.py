import sys
import time

import numpy                   as np 
import matplotlib.pyplot       as plt  

import subprocess

import tensorflow              as tf
from   tensorflow              import keras
from   tensorflow.keras        import layers, Input, backend
from   tensorflow.keras.layers import Dense, Activation, ZeroPadding2D, BatchNormalization, Conv2D, Permute
from   tensorflow.keras.layers import Lambda, Bidirectional, LSTM, GRU, Dot, Softmax
from   tensorflow.keras.models import Model

from   scipy.io                import wavfile
from   python_speech_features  import mfcc, logfbank 

from collections import Counter
import itertools

############
# Packages #
############

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])    



#############################################################################
#                          MODEL TRAINING FUNCTIONS                         #
#############################################################################

def reshape(X, y):
    
    """ 
    Given two associated numpy arrays, it reshapes them appropriately.
      
    Parameters: 
      X, y (numpy arrays): input data and associated label (or class)

    Returns: 
      X, y (numpy arrays): shuffled input data and associated label (or class) with an added dimension (for the channel)    

    """

    return np.expand_dims(X, axis = 3), np.expand_dims(y, axis = 1) 


def normalize(x, output_range = (-1,1), abs_norm = False, std = False):
    
    """
    This function applies a normalization of the features. By default does [-1,1]-Normalization, but it can be adapted to
    every desired range. 
    
    Parameters:
        
        x (numpy array): Feature data. Recall that 
        
        output_range (tuple): Tuple containing, respectively, the minimum value of the features after the 
        transformation and the maximum one (min_value, max_value) 
        
        abs_norm (bool): Boolean value. If True means that normalization is done extracting the min and the max values
        over all dimensions. 
        If False, then normalize w.r.t. the features (so it considers the individual variability of each mel-coefficient).
        The former does normalize in way that preserves the exact information content of the data meanwhile, the latter does normalize in way
        that gives the same relevance to each feature (mel-coefficient). 
        
        std (bool): Boolean value. If True does perform standardization with variance correction, otherwise performs standardization. 
        
    """
        
    if abs_norm:
    
        new_max, old_max = np.array([output_range[1],  np.max(x)])-5e-5 
        new_min, old_min = np.array([output_range[0],  np.min(x)])+5e-5
        
        return ((new_max-new_min)/(old_max-old_min))*(x-old_max) + new_max
    
    else: 
        
        old_max_array, old_min_array = np.max(np.max(x, axis = 2), axis = 0), np.min(np.min(x, axis = 2), axis = 0)
        old_max_matrix, old_min_matrix = np.outer(old_max_array, np.ones((1, x.shape[-1])))+5e-5 , np.outer(old_min_array, np.ones((1, x.shape[-1])))-5e-5
        new_max_matrix, new_min_matrix = np.ones((x.shape[-2], x.shape[-1]))*output_range[1], np.ones((x.shape[-2], x.shape[-1]))*output_range[0]
        
        return ((new_max_matrix-new_min_matrix)/(old_max_matrix-old_min_matrix))*(x-old_max_matrix) + new_max_matrix 


def plot_history(history, model_name, filename):

    """

    Plots accuracy/loss for training/validation set as a function of the epochs.

    Parameters:
      history: Training history of model.
      model_name: Name of the model.
      filename: Name of the file.

    """
    
    fig, axs = plt.subplots(2)
    
    # create accuracy subplot
    axs[0].plot(history.history["sparse_categorical_accuracy"], label="accuracy")
    axs[0].plot(history.history['val_sparse_categorical_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation of {}".format(model_name))
    
    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation of {}".format(model_name))
    
    plt.savefig(filename)
    
    plt.show()


def plot_confusion_matrix(cm, labels, title = "Confusion Matrix", normalize = False, display_metrics = True):

    """

    This function makes the plot of a given a sklearn confusion matrix (cm).
    
    Parameters: 
      cm (Sklearn cm): Confusion matrix from sklearn.metrics.confusion_matrix
      labels (dict): A dictionary containing the name of the categories and the corresponding labels
      normalize (bool): If False, plot the raw numbers; If True, plot the proportions
      display_metrics (bool): If True, displays model's metrics
    
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(f'{title}\n', fontsize=13)

    if labels is not None:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45, fontsize=12)
        plt.yticks(tick_marks, labels, fontsize=12)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 10 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            
        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()

    if display_metrics:
        plt.ylabel('True label', fontsize=10)
        plt.xlabel('\nPredicted label\naccuracy={:0.4f} - misclass={:0.4f}'.format(accuracy, misclass), fontsize=10)
    plt.show()


def wrong_predictions(model, X, y, filenames):
  
    """

    This function returns the samples incorrectly classified by the model.

    Parameters:
      model (TensorFlow model)
      X (ndarray): Contains the input features X 
      y (ndarray): Correct categories
      filenames (list): Contains the names of the files

    Returns:
      filenames_incorrect (list): Contains the file names of the samples incorrectly classified
      y_true (list): Contains the true labels
      y_pred (list): Contains the labels predicted by the model
      errors_per_category (Counter): Contains the number of samples incorrectly classified for each category

    """

    predictions = np.argmax(model.predict(X), axis=1)
    
    correct = (predictions == np.squeeze(y))
    incorrect = (correct == False)
    
    filenames_incorrect = filenames[incorrect]
    
    y_true = y[incorrect]
    y_pred = predictions[incorrect]

    count_errors =  []
    for i in range(len(filenames_incorrect)):
      count_errors.append(filenames_incorrect[i].split('/')[0])
      errors_per_category = Counter(count_errors)

    return filenames_incorrect, y_true, y_pred, errors_per_category

def rnn_architecture(name, input_shape, rnn, num_classes): 

    """

    Parameters:
      name (string): It holds a descriptive name of the model
      input_shape (tuple): It holds the dimensions of a single training example  
      rnn (string): It indicates the type of RNN to use in the model. It can be 'LSTM' or 'GRU'
      num_classes (int): It is the number of considered categories
    
    Returns: 
      model (TensorFlow model)
    
    """

    X_input = Input(input_shape, name = "input")  

    # PERMUTE: Mfccs have shape (batchsize, numcep, timesteps, 1) but LSTM and GRU require as input a 3D tensor with shape (batchsize, timesteps, numcep)
    X = Permute((2, 1, 3))(X_input) 

    # BATCH NORMALIZATION: 
    X = BatchNormalization(name='batch_norm0')(X)
    
    # CONV: Applied only in the time dimension to extract local relations in the audio file
    X = Conv2D(10, (5, 1), activation='relu', padding='same', name='conv0')(X)
      
    # BATCH NORMALIZATION: 
    X = BatchNormalization(name='batch_norm1')(X)
      
    # CONV:
    X = Conv2D(1, (5, 1), activation='relu', padding='same', name='conv1')(X)
      
    # BATCH NORMALIZATION: 
    X = BatchNormalization(name='batch_norm2')(X)

    # LAMBDA: Removes last dimension. Now X has shape (batchsize, timesteps, numcep)
    X = Lambda(lambda q: backend.squeeze(q, -1), name='squeeze_last_dim')(X)

    if rnn not in ['LSTM', 'GRU']:
          raise ValueError(
              'Invalid value for rnn parameter. Please enter LSTM or GRU.')

    # BIDIRECTIONAL LSTM: Bidirectional wrapper to LSTM layers in order to train the model in both directions (forward and backward).
    if rnn == 'LSTM':
      X = Bidirectional(LSTM(64, return_sequences=True), name='bidir_LSTM0')(X)  
      X = Bidirectional(LSTM(64), name='bidir_LSTM1')(X)

    # BIDIRECTIONAL GRU:
    if rnn == 'GRU':
      X = Bidirectional(GRU(64, return_sequences=True),  name='bidir_GRU1')(X)
      X = Bidirectional(GRU(64), name='bidir_GRU2')(X)

    # DENSE:
    X = Dense(64, activation='relu', name='fc0_relu')(X)
      
    X = Dense(32, activation='relu', name='fc1_relu')(X)

    X = Dense(num_classes, activation='softmax', name='fc2_softmax')(X)
    
    # Create the model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name = name)
      
    # Declaring the optimizer
    optimiser = tf.optimizers.Adam(learning_rate = 0.001)

    # compile model
    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics='sparse_categorical_accuracy')

    # print model parameters on console and strcuture
    model.summary()
      
    tf.keras.utils.plot_model(model, show_shapes=True)
      
    return model


def rnn_training(model, model_name, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):
    
    """

    Parameters:

      model (TensorFlow model)
      model_name: Name of the model
      epochs (int): Number of training epochs
      batch_size (int): Samples per batch
      patience (int): Number of epochs to wait before early stop, if there is not an improvement on accuracy
      X_train (ndarray): Inputs for the training set
      y_train (ndarray): Targets for the training set
      X_validation (ndarray): Inputs for the validation set
      y_validation (ndarray): Targets for the validation set
    
    Returns:
      history: Training history

    """

    # earlystop_callback 
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                     patience=patience,
                                                     verbose=1,
                                                     restore_best_weights=True)

    # Reduce Learning R on Plateau
    reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    patience=patience,
                                                    verbose=1)

    # Save best models
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint("checkpoint_{}.h5".format(model_name),
                                                         monitor='val_loss',
                                                         save_best_only=True)
    
   
    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
                        shuffle=True,
                        callbacks=[earlyStopping,modelCheckpoint, reduceLR])
    
    return history



def att_rnn_architecture(name, input_shape, num_classes):

    """

    Parameters:
        name (string): It holds a descriptive name of the model.
      input_shape (tuple): It holds the dimensions of a single training example.   
      num_classes (int): It is the number of considered categories.
    
    Returns: 
      model (TensorFlow model)
    
    """

    X_input = Input(input_shape, name = "input")  

    # PERMUTE: Mfccs have shape (batchsize, numcep, timesteps, 1) but LSTM requires as input a 3D tensor with shape (batchsize, timesteps, numcep)
    X = Permute((2, 1, 3))(X_input) 

    # BATCH NORMALIZATION: 
    X = BatchNormalization(name='batch_norm0')(X)
    
    # CONV: Applied only in the time dimension to extract local relations in the audio file
    X = Conv2D(10, (5, 1), activation='relu', padding='same', name='conv0')(X)
      
    # BATCH NORMALIZATION: 
    X = BatchNormalization(name='batch_norm1')(X)
      
    # CONV:
    X = Conv2D(1, (5, 1), activation='relu', padding='same', name='conv1')(X)
      
    # BATCH NORMALIZATION: 
    X = BatchNormalization(name='batch_norm2')(X)

    # LAMBDA: Removes last dimension. Now X has shape (batchsize, timesteps, numcep)
    X = Lambda(lambda q: backend.squeeze(q, -1), name='squeeze_last_dim')(X)

    # ATTENTION MECHANISM: Its implementation can be broken down into 3 steps.

    #STEP 1: Prepare hidden states. (we use every single hidden state generated by the LSTM layer)
    X = Bidirectional(LSTM(64, return_sequences=True), name='bidir_LSTM0')(X) #‘return_sequences=True’ returns the hidden state at each step
    X = Bidirectional(LSTM(64, return_sequences=True), name='bidir_LSTM1')(X)  

    # decide which hidden states are important 
    X_middle = Lambda(lambda q: q[:, -1])(X)  # extract the middle output vector of the last LSTM layer 
    X_query = Dense(128)(X_middle) # project the vector and used it as a query vector 

    # STEP 2: Obtain scores and attention weights
    attention_scores = Dot(axes=[1, 2])([X_query, X]) # dot product attention 
    attention_scores = Softmax(name='attention')(attention_scores)  # softmax layer

    # STEP 3: Multiply each hidden state by its softmax score.
    X_attention = Dot(axes=[1, 1])([attention_scores, X])  # weighted average of LSTM output 

    # DENSE:
    X = Dense(64, activation='relu')(X_attention)
    X = Dense(32)(X)

    X = Dense(num_classes, activation='softmax', name='output')(X)
    
    # Create the model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name = name)
      
    # Declaring the optimizer
    optimiser = tf.optimizers.Adam(learning_rate = 0.001)

    # compile model
    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics='sparse_categorical_accuracy')

    # print model parameters on console and strcuture
    model.summary()
      
    tf.keras.utils.plot_model(model, show_shapes=True)
      
    return model


def prediction_time(model, X_test):

  """

  This function computes the average of the time the model takes to predict one single test sample.

  Parameters:
    model (TensorFlow model)
    X_test (ndarray): array containing test samples
    
    """

  num_samples = len(X_test) # number of samples
  starts = np.empty((num_samples,))
  ends = np.empty((num_samples,))

  for i in range(num_samples):

    x = np.array([X_test[i]]) # takes one sample at time
    starts[i] = time.time() 
    a = np.argmax(model.predict(x),1)
    ends[i] = time.time()
 
  average_time = sum(ends-starts)/num_samples

  print("Average prediction time (ms):", average_time * 1000)