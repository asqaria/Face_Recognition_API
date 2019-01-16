from FRA import FaceRecgnitionApp
from imutils.video import VideoStream
import time


# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
stream = VideoStream('http://192.168.0.125:4747/video').start()
time.sleep(2.0)

# start the app
pba = FaceRecgnitionApp(stream)
pba.root.mainloop()
