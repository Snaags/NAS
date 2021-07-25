import numpy as np
import math
import os
import random
def window_array(x_arr,y_arr , window_size : int, step_size :int =1):
    """
    splits array into set of sequences of window_size.
    starting at x = 0 then x += step_size 
    dim 0 in x_arr should be the time dim
    """

    if x_arr.shape[0] != y_arr.shape[0]: 
        raise ValueError("X (length "+str(x_arr.shape)+") and Y (length "+str(y_arr.shape)+") arrays different lengths in time domain")

    #Define array sizes
    num_windows = math.ceil(len(x_arr)/step_size) - int((window_size -1)/ step_size )
    x_output = np.empty(( num_windows , window_size , *x_arr.shape[1:] ) , float )
    y_output = np.empty((num_windows),float)
    #Compute X Array
    for c in range(num_windows):
        x_output[c] = x_arr[c*step_size:(c*step_size)+window_size]
        y_output[c] = y_arr[(c*step_size)+window_size-1 ]
    return x_output, y_output
    
                    
def reduce_array_memory_usage(array, reduction : float):
    """
    removes entries from the array to reduce memory usage

    """
    if random.randint(0,1) == 1:
        return array[int(array.shape[0]*reduction):]

    else: 
        return array[:-int(array.shape[0]*reduction)]
    

  
def window_array_random(x_arr,y_arr , window_size : int, n_samples :int):

    """
    splits array into set of sequences of window_size.
    
    Randomly select windows of size window_size from a location in x_arr     

    x_arr should be a np array of shape (timesteps, features)
    """
    
    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
      
    # Memory usage
    print("RAM memory % used:", round((used_memory/total_memory) * 100, 2))
    windowed_array_x = list()
    windowed_array_y = list()
    for i in range(n_samples):
        random_index = random.randint(0,x_arr.shape[0]-window_size-1)
        window = x_arr[random_index:random_index+window_size]
        windowed_array_y.append(y_arr[random_index+window_size])
        windowed_array_x.append(window)
    print(len(windowed_array_x))
    return np.array(windowed_array_x, dtype = np.float32), np.array(windowed_array_y, dtype = np.float32)

 
        
if __name__ == "__main__":
    ##Tests 
    from TEPS_worker import load_dataset
    x,y ,x_t,y_t = load_dataset()
    x_out,y_out = window_array_random(x,y, 128,500000)    
        
