import os 
import sys
from datetime import date

import subprocess

import numpy as np 

from scipy.io import wavfile
from scipy import signal  
import librosa

try:
    from python_speech_features import mfcc, logfbank
except:
    install("python_speech_features")
    from python_speech_features import mfcc, logfbank

sys.path.append('../src/data')
import data_utils as ds

############
# Packages #
############

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package]) 



#############################################################################
#                        FEATURE EXTRACTION FUNCTIONS                       #
#############################################################################


def mfcc_feature_extraction(dataset_path, samples, json_path, json_name, numcep = 40, nfilt = 40, winlen = 0.025, winstep = 0.01, disposal = list(), verbosity = 0):

    """
    
    This function extracts the MFCCs representation for each .wav file stored in a given directory. 

    Parameters: 
        dataset_path (string): The relative path of the dataset  
        samples (int): Sampling frequency of each file to convert in mfcc 
        json_path (string): The relative destination path of the .json file 
        json_name (string): Name of the .json file, e.g. "file_name.json"
        numcep (int): Default is 40. Number of Mel's cepstral coefficients
        n_filt (int): The number of filters of the filterbank. Default is 40
        winlen (float): The length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        winstep (float): The step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        disposal (list): List with the filenames that are excludeed by the function 
        verbosity (int): Gives back additional info regarding the execution as the parameter increases {0,1,2}. Default is 0 
      
    Returns: 

        A .json file with the following keys: 
            "mappings": list of keywords corresponding the audio files
            "labels": numerical encoding of the keyword. E.g. the keyword "on" has label 0, "off" ha label 1
            "MFCCs": Mel Cepstral Coefficients associated with each file 
            "files": list of the audio filenames - dimension is 44 by 13 (number of cepstral coefficients)
            "encoding": dictionary containing the encoding of the categories
            
        The dictionary used to save the .json file. 
            Example of output. 
            >>>  data = {
                        "mappings":[0,1,2,3,4,5,6,7,8,9,...], 
                        "labels":[0,0,0,0,0,0,1,1,1,1,1,...],
                        "MFCCs":[...], 
                        "files":["dataset/on/1.wav",...]
                        "encoding":{"down":0,"go":1,...}
                        }
                
    """
    
    
    # creating the data dictionary
    data = {
    "mappings":[],
    "labels":[],
    "MFCCs":[],
    "files":[],
    "encoding":{}
    }

    # loop through all the sub-directories, i index is associated with the sub-directory
    for i,(dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # the walk method does pass also through the root directory. We need to avoid that   
        if dirpath is not dataset_path: 

            # update mappings
            
            category = dirpath.split("/")[-1] 
            # extracting the category from the dirname: "/dataset/down" -> ["dataset","down"][-1] -> ["down"]
            
            if category != "_background_noise_": 
                
                data["mappings"].append(category)
                data["encoding"][category] = i - 1


                if verbosity == 1: 
                    print(f"Processing the {category} category")

                # loop through all the filenames for each category to extract MFCCs
                for f in filenames: 
                    
                    complete_file_name = category + "/" + f
                    
                    if complete_file_name not in disposal: 
                        
                        # get the filepath (since we have only the filename)
                        file_path = dirpath + "/" + f # global filepath   

                        # saving the .wav file in a floating point time series format (signal) and the sampling rate (sr)
                        # signal, sr = librosa.load(file_path) 
                        sr, signal = wavfile.read(file_path)

                        # extract the MFCCs as a numpy ndarray - we used the same notation of the librosa method  
                        MFCCs = mfcc(signal, samplerate = samples, winlen = winlen, winstep = winstep, numcep = numcep, nfilt = nfilt) 
                        # store data 
                        data["labels"].append(i-1)
                        # since the os.walk method associates 
                        # 0 --> directory_path
                        # 1 --> sub_directory_1 ("no")
                        # 2 --> sub_directory_2 ("yes")
                        # ... 
                        # and we want to gain a label associated to each keyword that starts from 0, 
                        # we skipped the first index i (with the first conditional) and we assign 
                        # i-1 to each label, to have a 0 indexed labelling of the target variable

                        # store MFCCs

                        data["MFCCs"].append(MFCCs.T.tolist()) # we cast to a list to save it in a .json file (np object cannot be saved similarly)

                        # taking the id and the label of the recording
                        data["files"].append(category + "/" + file_path.split("/")[-1])

                        if verbosity == 2: 
                            print(f"{file_path}: {i-1}")

    ds.save_json(json_path, json_name, data)
    
    print("Executed on {}".format(date.today()))
          
    return data


def energy(signal, mean = False):
    
    """
    This function computes the discrete energy of a signal x(n), which by definition is E[x(n)] = Σ|x(n)|^2  for i = 1,..., N 

    Paramters: 
        signal (numpy array): The array containing  

    """

    shape = signal.shape[0]
     
    return np.sum(np.abs(signal)**2)/shape if mean else np.sum(np.abs(signal**2))


def short_time_energy(signal, hop_length = 256, frame_length = 512): 

    energy = np.array([
        np.sum(np.abs(signal[i:i+frame_length]**2))
        for i in range(0, len(signal), hop_length)
    ])
    
    return energy 

def zero_crossings(audio_dict, scaled = False, rate = True): 
    
    zero_crossings = {} 
    
    if rate:
        zero_crossing_rate = {}
    
    for category in audio_dict.keys():

        zero_crossings[category] = [] 
        
        if rate: 
            zero_crossing_rate[category] = []
        
        for file in audio_dict[category]:

            zc_value = librosa.zero_crossings(file)
            zero_crossings[category] += [np.sum(zc_value)]
            
            if rate: 
                
                zcr_value = librosa.feature.zero_crossing_rate(file.astype(np.float32)).T
                
                if scaled: 
                    zero_crossing_rate[category] += [zcr_value/file.shape[0]]
                else: 
                    zero_crossing_rate[category] += [zcr_value]
    
    print("Executed on {}".format(date.today()))
    
    if rate:
        return zero_crossings, zero_crossing_rate
    else: 
        return zero_crossings
    

def mean_energies(audio_dict): 
    
    mean_energies = {} 

    for category in audio_dict.keys():

        mean_energies[category] = [] 
        
        for file in audio_dict[category]:

            energy_value = energy(file, mean = True)
            mean_energies[category] += [np.sum(energy_value)]
                
    print("Executed on {}".format(date.today()))
    
    return mean_energies


def root_mean_square_energy(signal, hop_length = 256, frame_length = 512, short_time = True): 

    if short_time: 

        energy = np.array([
            np.sum(np.abs(signal[i:i+frame_length]**2))/frame_length
            for i in range(0, len(signal), hop_length)
        ])
        
    else: 
        
          energy = np.array([
            np.sum(np.abs(signal[i:i+frame_length]**2))
            for i in range(0, len(signal), hop_length)
        ])
        

    return np.sqrt(energy) 


def log_specgram(audio, sample_rate, window_size = 20, step_size = 10, eps = 1e-10):
    
    """
    This function calculates log-spectrograms. Since human do not hear loudness on a linear scale, 
    it takes the logarithm of the spectrogram values.

    Parameters:
        audio:(numpy array): The audio files on which the spectrogram is computed
        sample_rate (int): Sample rate of .wav file
        window_size (int): It is the size of the window on which the Fourier Transform is computed. Default is 20
        step_size (int): It is the number of points to ovelap between segments. Default is 10
        eps (int): It is a number to calculate the logarithm of the spectrogram. Default is 1e-10
    
    Returns:
        freqs (numpy array): Array of sample frequencies
        times (numpy array): Array of segment times
        log_spec (numpy array): Log-Spectrogram of the audio
    
    """
    
    # length of each segment
    nperseg = int(round(window_size * sample_rate / 1e3)) # division by 1e3 because window is 20 ms
    # number of points to overlap between segments
    noverlap = int(round(step_size * sample_rate / 1e3)) 
    # computing the spectrogram
    freqs, times, spec = signal.spectrogram(audio, fs = sample_rate, window = 'hann', nperseg = nperseg, noverlap = noverlap, detrend=False)
    # taking the logarithm 
    log_spec = np.log(spec.T.astype(np.float32) + eps)
    
    return freqs, times, log_spec



