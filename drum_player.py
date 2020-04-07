# organize imports
import cv2
import sound_player

def main():
    player = sound_player.SoundPlayer()
    cap = cv2.VideoCapture(0)

    while(True):
        _, frame = cap.read()
        cv2.imshow('Frame', frame)
        player.on_key_pressed(cv2.waitKey(1))
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()