import simpleaudio as sa
import threading
import time
import datetime

KEY_DISTRIBUTION_FILE = 'sounds/notes/key_distribution.txt'
RECORDINGS_DIRECTORY = 'recordings'
START_NOTE = -1

def play_file(filename):
        wave_obj = sa.WaveObject.from_wave_file(filename)
        wave_obj.play()

class SoundPlayer(object):
    def __init__(self):
        self.threads = []
        self.key_map = {}
        self.recording = [[time.time(), START_NOTE]]

        lines = open(KEY_DISTRIBUTION_FILE, 'r').readlines()
        for line in lines:
            key_sound = line.split(' ')
            key, sound = key_sound[0], key_sound[1][:-1]
            self.key_map[ord(key)] = f'sounds/{sound}'
    
    # Comment test
    def play_key(self, key):
        if not key in self.key_map.keys():
            return

        self.recording.append([time.time(), key])
        self.play_file_on_separate_thread(self.key_map[key])
    
    def print_logs(self):
        print(self.recording)

    def get_sound_for_key(self, key):
        return self.key_map[key]

    def play_file_on_separate_thread(self, filename):
        thread= threading.Thread(target=play_file, args=(filename,))
        thread.start()
        self.threads.append(thread)

    def play_file_and_wait(self, filename):
        wave_obj = sa.WaveObject.from_wave_file(filename)
        play_obj = wave_obj.play()
        play_obj.wait_done() 
    
    def clear_recording(self):
        self.recording = []

    def play_recording_util(self, recording_to_play, play_from_first_note=False):
        recording_copy = []
        for entry in recording_to_play:
            recording_copy.append(entry.copy())

        if play_from_first_note:
            recording_copy.pop(0)

        recording_start_time = recording_copy[0][0]
        for i in range(len(recording_copy)):
            recording_copy[i][0] -= recording_start_time
        
        playing_start_time = time.time()
        ind = 0
        while ind < len(recording_copy):
            if (time.time() - playing_start_time) > recording_copy[ind][0]:
                self.play_key(recording_copy[ind][1])
                ind += 1
        
    def play_recording(self, play_from_first_note=False):
        self.play_recording_util(self.recording, play_from_first_note=play_from_first_note)
    
    def save_recording(self, file_name=' '):
        if file_name == ' ':
            file_name = str(f'recording_at_{datetime.datetime.now()}')
        
        file = open(f'{RECORDINGS_DIRECTORY}/{file_name}', 'w')

        for entry in self.recording:
            file.write(f'{entry[0]} {entry[1]}\n')
        
        file.close()
    
    def play_file(self, file_name):
        lines = open(file_name, 'r').readlines()
        recording = []
        for line in lines:
            time_note = line.split(' ')
            time, note = time_note[0], time_note[1][:-1]
            recording.append([float(time), int(note)])
        
        self.play_recording_util(recording)

    
