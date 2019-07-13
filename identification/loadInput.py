import numpy as np
from identification import utils

def load_txt(path):

    # normalize path
    path = utils.normpath(path)

    # read file line by line
    with open(path, 'rb') as fid:
        lines = fid.readlines()

    values = []   # values in the input file
    for item in lines:
            values.append(item)

    sampling_rate = 1000
    resolution = 12

    # convert mdata
    mdata = {}   # a dictionary to hold the values
    df = '%Y-%m-%dT%H:%M:%S.%f'
    try:
        mdata['sampling_rate'] = float(sampling_rate)
    except KeyError:
        pass
    try:
        mdata['resolution'] = int(resolution)
    except KeyError:
        pass

    # load array
    data = np.genfromtxt(values, delimiter=b',')

    print('data',data)

    return data, mdata