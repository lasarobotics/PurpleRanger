import threading

global video_lock
global video_frame
global baseline

video_lock = threading.Lock()
video_frame = None

baseline = 0.075