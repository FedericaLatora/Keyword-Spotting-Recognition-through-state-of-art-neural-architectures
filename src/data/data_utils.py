import os 
import sys
import json  
from   datetime                import date
  
import subprocess

from random import sample
import numpy                   as np 
import matplotlib.pyplot       as plt  

from   tensorflow.keras.models import Model

from   scipy.io                import wavfile
from   python_speech_features  import mfcc, logfbank
import wave 

import subprocess
from IPython.display            import Audio

try:
    from python_speech_features import mfcc, logfbank
except:
    install("python_speech_features")
    from python_speech_features import mfcc, logfbank

############
# Packages #
############

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])    



#############################################################################
#                         DATA MANIPULATION FUNCTIONS                       #
#############################################################################
    

def stratified_sampling(X, y, num_classes, ratio = 0.4):
    
    """
   
    This function extracts a stratified sample of two associated numpy arrays.
    
    Parameters:  
        X, y (numpy arrays): input data and associated label (or class)
        num_classes (int): number of classes 
        ratio (float): ratio of the original data to be sampled. Default is 0.4
        
    Returns:
        X_sample, y_sample (numpy arrays): sampled input data and associated label (or class).
        
    """
    
    for i in range(num_classes):
        
        # generating sample indexes
        indexes = np.random.randint(low = 0, 
                                    high = y[y == i].shape[0], 
                                    size = int(y[y == i].shape[0]*ratio))
        if i == 0: 
            X_sample, y_sample = X[y == i][indexes], y[y == i][indexes]
        else: 
            X_sample  = np.concatenate((X_sample, X[y == i][indexes]))
            y_sample = np.concatenate((y_sample, y[y == i][indexes])) 
            
    print("Stratified samples of X, y with ratio {} \n original shape {} \n sampled shape {} ".format(ratio,  y.shape[0], y_sample.shape[0]))
    
    return X_sample, y_sample


def extract_audio(dataset_path, documentation_filename, extract_sr = False, json_path = None, json_name = None, save = True):
   
    """
    
    Given the desired amount of files per category this function extracts a dictionary 
    of audio files' signals and their corresponding name, by random selection.

    Parameters: 
        dataset_path (string): The relative path of the dataset
        documentation_filename (string): The relative path of the file needed for the splitting
        extract_sr (bool): If True, extracts the sampling rate from the audio
        json_path (string): Absolute or Relative path of the json 
        json_name (string): Name of the json
        save (bool): Save the output in a json file called <json_name>

    Returns:
        audio_dict (dict): An example of the output is:
                           {"category_1": ([np.array([0.3839,-0.192388,....])], "category_1/audio_filename_000.wav"), ... }
    
    """
    
    # creating dictionary of files with categories as keys
    filenames_dict = {}
    
    # importing the list of filenames from the .txt document 
    filenames = [file[:-1] for file in open(documentation_filename, "r").readlines()]
    
    # populating the dictionary
    for file in filenames: 
        
        # extracting category
        category = file.split("/")[0]
        
        # checking if category is already a key
        if category in filenames_dict.keys():
            
            # if yes then add the file to the related category/key
            filenames_dict[category] += [file]
        else:
            # if not then create a list with the file inside
            filenames_dict[category] = [file]
    
    # creating output dict
    audio_dict = {}
    
    # creating the dict with the correspondent audio 
    for category in filenames_dict.keys():
        
        audio_dict[category] = []
        
        for file in filenames_dict[category]:
        
            file_path = dataset_path + file
            audio_signal = wavfile.read(file_path) if extract_sr else wavfile.read(file_path)[1]
            audio_dict[category] += [audio_signal]
    
    
    if save == True and not json_name and not json_path: 
        
        save_json(json_path, json_name, audio_dict)
    
    print("Executed on {}".format(date.today()))
    
    return audio_dict, filenames_dict


def extract_random_audio(dataset_path, documentation_filename, extract_sr = False, json_path = None, json_name = None, file_amount = 10, save = True):
    
    """
    
    Given the desired amount of files per category this function extracts a dictionary 
    of audio files' signals and their corresponding name, by random selection.

    Parameters: 
        dataset_path (string): The relative path of the dataset
        documentation_filename (string): The relative path of the file needed for the splitting
        extract_sr (bool): If True, extracts the sampling rate from the audio
        json_path (string): Absolute or Relative path of the json 
        json_name (string): Name of the json
        file_amount (int): Desidered number of files per category 
        save (bool): Save the output in a json file called <json_name>

    Returns:
        audio_dict (dict): An example of the output is:
                           {"category_1": ([np.array([0.3839,-0.192388,....])], "category_1/audio_filename_000.wav"), ... }
    
    """
    
    # creating dictionary of files with categories as keys
    filenames_dict = {}
    
    # importing the list of filenames from the .txt document 
    filenames = [file[:-1] for file in open(documentation_filename, "r").readlines()]
    
    # populating the dictionary
    for file in filenames: 
        
        # extracting category
        category = file.split("/")[0]
        
        # checking if category is already a key
        if category in filenames_dict.keys():
            
            # if yes then add the file to the related category/key
            filenames_dict[category] += [file]
        else:
            # if not then create a list with the file inside
            filenames_dict[category] = [file]
    
    # creating new dict for randomly sampled data 
    random_dict = {}
    
    # selection of random elements from filenames_dict
    for category in filenames_dict.keys():
        
        random_dict[category] = sample(filenames_dict[category], file_amount)
    
    # creating output dict
    audio_dict = {}
    
    # creating the dict with the correspondent audio 
    for category in random_dict.keys():
        
        audio_dict[category]= []
        
        for file in random_dict[category]:
        
            file_path = dataset_path + file
            audio_signal = wavfile.read(file_path) if extract_sr else wavfile.read(file_path)[1]
            audio_dict[category] += [audio_signal]
    
    
    if save == True and not json_name and not json_path: 
        
        save_json(json_path, json_name, audio_dict)
    
    print("Executed on {}".format(date.today()))
    
    return audio_dict


def shuffle(X, y):
    
    """ 
    
    Given two associated numpy arrays, it shuffles them randomly.

    Parameters: 
        X, y (numpy arrays): input data and associated label (or class)

    Returns: 
        X, y (numpy arrays): shuffled input data and associated label (or class) with an added dimension (for the channel)    
    
    """

    shuffle_idx = np.arange(y.shape[0])
    shuffle_rng = np.random.RandomState(123)
    shuffle_rng.shuffle(shuffle_idx)

    return X[shuffle_idx], y[shuffle_idx]
  
    
def save_json(location, name, dictionary): 
    
    """
    
    This function saves a dictionary in a .json format.
    The name must include the extension: "file_name.json"
    
    Parameters: 
    
        location (string): Absolute or Relative path. Saving location of the json file
        name (string): Name of the json file 
        dictionary (dict): Dictionary to save
        
    """

    with open(location + name, "w") as file: 
        json.dump(dictionary, file, indent = 4)

        
def import_data_dictionary(json_path):
    
    """
   
    This function imports the json file stored at <json_path> and gives back its content as a dict object.
    Ensure to have the json module installed and imported in your environment. 
    
    Parameters:  
        json_path (string): Absolute or Relative path of the imported json 
        
    Returns:
        data_dictionary (dict): A dictionary containing all the information stored in the json file
        
    """
    
    with open(json_path, "r") as file:
        data_dictionary = dict(json.load(file))

    return data_dictionary


def intersection(list1, list2):
    
    """
    
    This function, given 2 lists, returns the list of values in common and 
    the indexes of the values of list1 belonging to the overlapping. 
    
    Parameters: 
        list1, list2 (list)
        
    Returns: 
        list3 (list): A list containing all the elements in common between list1 and list2
        indexes (list): A list containing all the indexes of the elements in common between the two list respect to list1
        
    """
    
    list3 = [value for value in list1 if value in list2]
    indexes = [list1.index(value) for value in list1 if value in list2]
    
    if len(list2) == len(list3):
        return list3, indexes
    else: 
        print("The length of the filter list it's not equal to the length of the overlapping list")
    

def extract_set_list_from_text(path, filename, reduced_classes = True):
    
    """
    
    This function generates a list and a set of strings containing the names of the files 
    in the text file <filename> stored at <path>, that belong to the classes specified
    by the <reduced_classes> parameter.
    
    Parameters: 
        path (string): Relative path where is stored the text file from which extract the list and the set
        filename (string): Name of the file of interest
        reduced_classes (bool): If `True`, then selects only files that belong to the 10 classes: 
                                'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'
                                Otherwise the function is applied on all the 35 classes in the data.
                                Default is `True`
    
    Returns: 
        filename_list (list): list containing all elements of the file 
        filename_set (set): set containing all elements of the file, without replication
        
    """
    
    filename_list = []
    filename_set = set()
    
    if reduced_classes: 

        for line in open(path + filename, "r"):

            if line.split("/")[0] in ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']:
                
                filename_list = filename_list + [line[:-1]] 
                filename_set.add(line[:-1])
    else: 
        
        for line in open(path + filename, "r"):

            if line.split("/")[0] in ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']:
                    
                filename_list = filename_list + [line[:-1]]  
                filename_set.add(line[:-1])
            
    return filename_list, filename_set


def fast_default_data_splitting(data_path, documentation_path, disposal = list(), reduced_classes = True):
    
    """
    
    This function splits the json file stored at <data_path> accordingly to the paper 
    "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition" in train-validation-test, 
    as defined in the text files stored at <documentation_path>.
 
    Parameters: 
        data_path (string): Absolute or relative path where is stored the json
        documentation_path (string): Absolute or relative path where are stored the files needed for the splitting
        disposal(list): List with the filenames that are excludeed by the function 
        reduced_classes (bool): If `True`, then selects only files that belong to the 10 classes: 
                                'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'
                                Otherwise the function is applied on all the 35 classes in the data.
                                Default is `True`
        
    Returns: 
        X_train, y_train, X_valid, y_valid, X_test, y_test (numpy arrays): arrays containing the input features X and the corresponding label y, 
                                                                           split in train-validation-test
        (train_filenames, valid_filenames, test_filenames) (tuple of lists): a tuple of lists containing the names of the files in the exact order of each split 
                                                                           (train, validation, testing) 
        
    """
    
    data = import_data_dictionary(json_path = data_path) # import json file 
    
    if reduced_classes: 
        
        # Creating the splitting lists for validation and test:  
        validation_list = extract_set_list_from_text(path = documentation_path, filename = "validation_list.txt")[0]
        test_list =       extract_set_list_from_text(path = documentation_path, filename = "testing_list.txt")[0]

    else: 
        
        # Creating the splitting lists for validation and test:  
        validation_list = extract_set_list_from_text(path = documentation_path, filename = "validation_list.txt", reduced_classes = False)[0]
        test_list =       extract_set_list_from_text(path = documentation_path, filename = "testing_list.txt",    reduced_classes = False)[0]        

    if disposal:
        
        validation_list = list( set(validation_list).difference(disposal) )
        test_list =       list( set(test_list).difference(disposal) )
                          
        validation_indexes =  intersection(data["files"], validation_list)[1]
        test_indexes =        intersection(data["files"], test_list)[1]
        
        training_indexes = sorted(list(set(range(0,len(data["files"]))).difference(validation_indexes).difference(test_indexes).difference(disposal)))
        
    else:
        
        # Extracting the indexes: 
        validation_indexes =  intersection(data["files"], validation_list)[1]
        test_indexes =        intersection(data["files"], test_list)[1]
        training_indexes = sorted(list(set(range(0,len(data["files"]))).difference(validation_indexes).difference(test_indexes)))

    MFCCs = np.array(data["MFCCs"])
    labels = np.array(data["labels"])
    filenames = data["files"]

    del data 

    X_train, y_train, train_filenames = MFCCs[training_indexes],   labels[training_indexes],   [filenames[index] for index in training_indexes]
    X_valid, y_valid, valid_filenames = MFCCs[validation_indexes], labels[validation_indexes], [filenames[index] for index in validation_indexes]
    X_test,  y_test,  test_filenames  = MFCCs[test_indexes],       labels[test_indexes],       [filenames[index] for index in test_indexes]

    return X_train, y_train, X_valid, y_valid, X_test, y_test, (train_filenames, valid_filenames, test_filenames)
    
    
def length_sanity_check(dataset_path, sampling_rate = 16000, verbosity = 0):
    
    """
    
    This function check if the length of each audio .wav file in the <dataset_path> folder is 1s long.  
    Returns the list containing such short audio files, their distribution across categories, 
    the distribution of all samples wrt categories and the total number of samples.  
     It returns the list of names of non com.
    
    Parameters:
        dataset_path (string): The relative path of the dataset
        sampling_rate (int): The sampling rate at which is processed each .wav file for 1s long audio file. Default is 16000
        verbosity (int): Gives back additional info regarding the execution as the parameter increases {0,1,2}. Default is 0

    Returns:
        short_waves (list): Contains short audio files' paths
        short_waves_distribution (dict): Contains the distribution of short samples wrt categories
        waves_distribution (dict): Contains the distribution of all samples wrt categories
        counter (int): Total number of samples

    """
    
    if verbosity == 2:
        
        if input("Are you sure to put verbosity = 2? It could crash your I/O interface. Accepted replies y/n:    ").lower() == "n": 
            verbosity = int(input("What value do you prefer? Accepted replies 0/1/2:     "))

    short_waves_distribution = {}
    waves_distribution = {}
    short_waves = []
    counter = 1
    
    for (dirpath, dirnames, filenames) in os.walk(dataset_path):

            if dirpath is not dataset_path:
                
                # extracting the category from the dirname: "/dataset/down" -> ["dataset","down"][-1] -> ["down"]
                category = dirpath.split("/")[-1] 

                if category != "_background_noise_": 
                    
                    # User info content display 
                    if verbosity == 1: 
                        print(f"Processing the {category} category")
                        
                    # Initalizazing counter for all dicts
                    short_waves_distribution[category] = 1
                    waves_distribution[category] = 1 
                    
                    # Iterating over all files of a given category
                    for f in filenames: 
                        
                        # User info content display
                        if verbosity == 2: 
                            print(f"Processing {f} file")
                            
                        # Update counter for each category and for whole dataset 
                        waves_distribution[category] += 1
                        counter += 1
                        
                        # Unpacking sampling rate and numpy array of amplitude values of time series' audio
                        sampling_rate, wave = wavfile.read(os.path.join(dirpath,f))

                        if len(wave) != sampling_rate: 
                            
                            short_waves_distribution[category] += 1
                            short_waves.append(category + "/" + f)
                    
    print("Files shorter then 1 s: ", len(short_waves))
    
    return short_waves, short_waves_distribution, waves_distribution, counter


def add_random_noise(dataset_path):    

    """
    
    This function searches for audio files in the dataset that contain less than 16000 samples and adds to them random noise. 
    At the end, the dataset will be entirelly composed by audio with 16000 samples.
    
    Parameters:
        dataset_path (string): The relative path of the dataset  
        
    """
    
    #Loop through all the sub-directories
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        
    #Loop through all the filenames for each category
        for f in filenames:
        
            # get the filepath (since we have only the filename)
            file_path = dirpath + '/' + f # global filepath         
        
            # saving the .wav file in a floating point time series format (signal) and the sampling rate (sr)
            sr, signal = wavfile.read(file_path) 
            
            if len(signal)<16000:
                
                # adding random noise
                random_noise = np.random.randint(-30, 30, 16000-len(signal))
                signal = np.append(np.asarray(signal), random_noise)
                
                # overwrite the original bad signal with the version padded with random noise
                wavfile.write(file_path, 16000, signal.astype(np.int16))  


def envelope_coefficients(x):

    """

    This function extract 13 coefficients from the mfccs.

    Parameters:
        x (ndarray): Contains mfccs

    Returns:
        The first 13 mfccs

    """

    return x[:, :13, :]


def set_dark_theme(figsize=(8, 8), dpi=100):

    """
    This function set the dark theme for the notebook and a default size for the figures. 

    """

    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)


def plot_audio(dataset_path, file_name, y_true, y_pred):

    """

    This function plots the raw waveform of a given audio that has been incorrectly classified by the model.
  
    Parameters:
        dataset_path (string): Relative path of the dataset
        file_name (string): Name of the audio file
        y_true (string): Correct audio's category
        y_pred (string): Audio's category predicted by the model

    """
  
    sr, wave = wavfile.read(dataset_path+file_name)

    plt.figure(figsize=(8,2))
    plt.plot(wave)
    plt.title('Raw waveform of '+file_name+' ($\mathregular{y_{true}}$=' + y_true+ ', $\mathregular{y_{pred}}$='+y_pred+')')
    plt.xlabel('Sample index')
    plt.ylabel('Amplitude')
  
    plt.tight_layout()

    return Audio(wave, rate = sr)


def plot_attention(file_name, model):

    """

    This function plots the raw waveform and the corresponding attention weights of an audio file. 

    Parameters:
        file_name (string): Name of the audio file
        model (TensorFlow model)

    """ 
  
    sr, wave = wavfile.read(file_name)
    mfccs = mfcc(wave, samplerate = sr, winlen = 0.025, winstep = 0.01, numcep = 13, nfilt=40) 
    mfccs = np.array(mfccs).T
    mfccs = np.expand_dims(mfccs, axis=2)
    X_example = np.expand_dims(mfccs, axis=0)

    attention_model = Model(inputs=model.input, outputs=[model.get_layer('attention').output])
    X_att = attention_model.predict(X_example)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10,4))
    ax1.plot(wave)
    ax1.set_title('Raw waveform of {}'.format(file_name))
    ax1.set_xlabel('Sample index')
    ax1.set_ylabel('Amplitude')
    ax2.plot(np.log(X_att.T))
    ax2.set_title('Attention weights (log)')
    ax2.set_xlabel('Mel-spectrogram index')
    ax2.set_ylabel('Log of attention weight')
    plt.tight_layout()

    return Audio(wave, rate = sr)


def plots_attention_dataset(dataset_path, num_classes, labels, model, num_plots):

    """

    This function plots the raw waveform and the corresponding attention weights of two audio files. In particular, it selects the first audio file 
    for each category's folder.

    Parameters:
        dataset_path (string): Relative path of the dataset
        num_classes (int): Number of categories
        labels (dict): Contains categories' names and corresponding numerical labels
        model (TensorFlow model)
        num_plots (int): Number of output plots

    """

    # extract the first audio file from each cateogry
    file_names = []
    file_paths = []
    for root, dirs, files in os.walk(dataset_path):
        dirs.sort()
        if files:
            file_names.append(sorted(files)[0])

    for i in range(0,num_classes):
        file_paths.append(dataset_path + list(labels.keys())[list(labels.values()).index(i)] + '/' +  file_names[i])  

    fig, (ax1, ax2) = plt.subplots(2,2,figsize=(24,9))
    for i in range(0,2):
        f = file_paths[i+num_plots]
        sr, wave = wavfile.read(f)
        mfccs = mfcc(wave, samplerate = sr, winlen = 0.025, winstep = 0.01, numcep = 13, nfilt=40) 
        mfccs = np.array(mfccs).T
        mfccs = np.expand_dims(mfccs, axis=2)
        X_example = np.expand_dims(mfccs, axis=0)

        attention_model = Model(inputs=model.input, outputs=[model.get_layer('attention').output])
        X_att = attention_model.predict(X_example)

        ax1[i].plot(wave)
        ax1[i].set_title('Raw waveform of {}'.format(f))
        ax1[i].set_xlabel('Sample index')
        ax1[i].set_ylabel('Amplitude')
        ax2[i].plot(np.log(X_att.T))
        ax2[i].set_title('Attention weights (log)')
        ax2[i].set_xlabel('Mel-spectrogram index')
        ax2[i].set_ylabel('Log of attention weight')
    fig.show()

    return