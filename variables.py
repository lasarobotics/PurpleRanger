import threading

global video_lock
global video_frame

video_lock = threading.Lock()
video_frame = None