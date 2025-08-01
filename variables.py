import threading

global video_lock
global video_frame
global baseline
global trace

video_lock = threading.Lock()
video_frame = None

baseline = 0.075

trace = False

main_pid = 0