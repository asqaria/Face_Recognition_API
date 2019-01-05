from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import Tkinter as tki
import threading
from pygame import mixer
import cv2
import face_recognition
import serial
import requests


class FaceRecgnitionApp:
    def __init__(self, stream):
        # store the video stream object, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.stream = stream
        self.frame = None
        self.thread = None
        self.stopEvent = None

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None
        mixer.init()
        # init
        self.ctn = True
        self.ctn_i = 0
        self.ser = serial.Serial('/dev/ttyACM0')
        self.ampl = 0

        self.state = False
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.end_fullscreen)

        # User info
        self.user_info = tki.Frame(self.root, padx=10, pady=150, height=500, width=500)

        name = tki.Label(self.user_info, pady=50, text="YOUR NAME HERE", font=("Helvetica", 26))
        name.pack(side='bottom')

        picture = tki.Canvas(self.user_info, width=400, height=300)
        picture.pack(side='top')

        self.img = ImageTk.PhotoImage(Image.open("default.png"))
        picture.create_image(20, 20, anchor=tki.NW, image=self.img)

        self.user_info.pack(side="right", fill="both", expand=True)

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.video_loop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_close)

    def video_loop(self):
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                # grab the frame from the video stream and resize it to
                # have a maximum width of 300 pixels
                self.frame = self.stream.read()

                face_locations = face_recognition.face_locations(self.frame)
                for (top, right, bottom, left) in face_locations:
                    self.ampl = right - left
                    if self.ampl < 130:
                        print('Far distance: %s' % self.ampl)
                        continue
                    elif self.ctn:
                        print('Scanning')
                        imgname = "frame.jpg"
                        cv2.imwrite(imgname, self.frame)
                        images = {'images': open(imgname, 'rb')}
                        r = requests.post('http://localhost:3434/recognize/', files=images)

                        j = r.json()
                        if j['id'] == 'Null':
                            print("Not recognized")
                            mixer.music.load("2.wav")
                        else:
                            print("Welcome %s (%.2f)" % (j['name'], j['confidence']))
                            self.ser.write(b'0')
                            mixer.music.load("1.wav")
                        mixer.music.play()

                        self.ctn = False
                        self.ctn_i = 0

                    cv2.rectangle(self.frame, (left, top), (right, bottom), (255, 0, 0), 1)

                if self.ctn_i > 50:
                    self.ctn = True
                else:
                    self.ctn_i += 1

                # OpenCV represents images in BGR order; however PIL
                # represents images in RGB order, so we need to swap
                # the channels, then convert to PIL and ImageTk format
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                # if the panel is not None, we need to initialize it
                if self.panel is None:
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)

                # otherwise, simply update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

        except RuntimeError, e:
            print("[INFO] caught a RuntimeError")

    def on_close(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.stream.stop()
        self.root.quit()

    def toggle_fullscreen(self):
        self.state = not self.state  # Just toggling the boolean
        self.root.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self):
        self.state = False
        self.root.attributes("-fullscreen", False)
        return "break"
