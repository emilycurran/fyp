import matplotlib.pyplot as plt
import wfdb
import numpy as np

record_name = '/home/emily/fyp/fyp/fyp/processing_data/mit-bih-arrhythmia-database-1.0.0/230'
signal, fields = wfdb.rdsamp(record_name)
annotations = wfdb.rdann(record_name, 'atr')
print(type(annotations))

def segmentation(records):
    Normal = []
    for e in records:
        signals, fields = wfdb.rdsamp(e, channels = [0]) 

        ann = wfdb.rdann(e, 'atr')
        good = ['N']
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        beats = (ann.sample)
        for i in imp_beats:
            beats = list(beats)
            j = beats.index(i)
            if(j!=0 and j!=(len(beats)-1)):
                x = beats[j-1]
                y = beats[j+1]
                diff1 = abs(x - beats[j])//2
                diff2 = abs(y - beats[j])//2
                Normal.append(signals[beats[j] - diff1: beats[j] + diff2, 0])    
    return Normal

normals  = segmentation(['/home/emily/fyp/fyp/fyp/processing_data/mit-bih-arrhythmia-database-1.0.0/230'])

print(normals[0])

# Plot a segment of the ECG signal with annotations
plt.plot(normals[1])  # Assuming single-channel ECG data, otherwise select the desired channel
plt.title('ECG Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

# Mark annotation points on the plot
plt.legend()
plt.show()