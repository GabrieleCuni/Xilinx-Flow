import numpy as np

data = np.load('./PreprocessCalibImages/calib_data.npz')['data']

batch_size=32

def calib_input(iter):

    calib_data = data[iter*batch_size:(iter+1)*batch_size]

    return {'input': calib_data}