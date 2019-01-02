import face_recognition
import cv2
import requests
from pygame import mixer
import serial

# init pygame to play sounds
mixer.init()
ser = serial.Serial('/dev/ttyACM0')

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture('http://192.168.0.197:4747/video')

# Initialize some variables
face_locations = []

# init
ctn = True
ctn_i = 0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # video_capture.release()
    face_locations = face_recognition.face_locations(rgb_small_frame)

    # Display the results
    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        ampl = right-left

        if ampl < 130:
            print('too far')
            continue
        elif ctn:
            print('Scanning')
            imgname = "frame%d.jpg" % ret
            cv2.imwrite(imgname, frame)
            images = {'images': open(imgname, 'rb')}
            r = requests.post('http://localhost:3434/recognize/', files=images)

            j = r.json()
            if j['id'] == 'Null':
                print("Not recognized")
                mixer.music.load("2.wav")
            else:
                print("Welcome %s" % j['name'])
                ser.write(b'0')
                mixer.music.load("1.wav")
            mixer.music.play()

            ctn = False
            ctn_i = 0

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 1)

    if ctn_i > 100:
        ctn = True
    else:
        ctn_i += 1

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
