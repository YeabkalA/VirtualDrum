from scipy.io import wavfile

fs, data = wavfile.read('sounds/drum-machine-snare.wav')
print(fs)
print('--------')
print(data)
print(len(data))

# for d in data:
#     print(d)