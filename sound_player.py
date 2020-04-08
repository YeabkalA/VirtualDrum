import simpleaudio as sa
import threading
import time

KEY_DISTRIBUTION_FILE = 'sounds/notes/key_distribution.txt'

def play_file(filename):
        wave_obj = sa.WaveObject.from_wave_file(filename)
        wave_obj.play()

class SoundPlayer(object):
    def __init__(self):
        self.threads = []
        self.key_map = {}
        self.recording = []
        keys_file = open(KEY_DISTRIBUTION_FILE, 'r')
        lines = keys_file.readlines()

        for line in lines:
            key_sound = line.split(' ')
            key, sound = key_sound[0], key_sound[1][:-1]
            self.key_map[ord(key)] = f'sounds/{sound}'
    
    def on_key_pressed(self, key):
        if not key in self.key_map.keys():
            return

        self.recording.append((time.time(), key))
        self.play_file_on_separate_thread(self.key_map[key])

        
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
    
