# organize imports
import cv2
import sound_player

EXIT_KEY = 'q'

class DrumPlayer(object):
    def __init__(self):
        self.player = sound_player.SoundPlayer()
    
    def run(self):
        cap = cv2.VideoCapture(0)
        while(True):
            _, frame = cap.read()
            cv2.imshow('Frame', frame)

            key = cv2.waitKey(1)
            if key == ord(EXIT_KEY):
                self.player.print_logs()
                self.player.play_recording(play_from_first_note=True)
                self.player.save_recording('test')
                return
            self.player.play_key(key)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # drum_player = DrumPlayer()
    # drum_player.run()
    sound_player.SoundPlayer().play_file('recordings/test')
