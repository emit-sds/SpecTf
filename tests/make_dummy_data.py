import h5py
import numpy as np
import csv
import os

NUM_BANDS = 200
NUM_DATAPOINTS = 100
NUM_CLASSES = 2

base = os.path.dirname(__file__)

if __name__ == "__main__":
    with h5py.File(os.path.join(base, 'data/mock_dataset.hdf5'), 'w') as f:
        f.create_dataset('labels', data=np.random.randint(0, NUM_CLASSES, (NUM_DATAPOINTS,)).astype(np.float32))
        fid_arr = np.array([
            np.random.choice(['train', 'test']).encode('utf-8')
            for _ in range(NUM_DATAPOINTS)
        ], dtype='S5')
        f.create_dataset('fids', data=fid_arr)
        f.create_dataset('spectra', data=np.random.rand(NUM_DATAPOINTS, NUM_BANDS))
        f.attrs['bands'] = np.arange(NUM_BANDS)

    def write_csv(n:str) -> None:
        with open(os.path.join(base, 'data/mock_'+n+'.csv'), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([n])
    
    write_csv('train')
    write_csv('test')