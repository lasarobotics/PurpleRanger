#!/usr/bin/env python

import os
import sys
import time
import json
import signal
import logging
import argparse
import threading

import cv2

import ntcore
from wpiutil import wpistruct
from wpimath.geometry import Pose3d, Rotation3d, Quaternion

from flask import Response
from flask import Flask
from flask import render_template

import variables
from pipelines.pipeline import Pipeline
from pipelines.basalt import Basalt
from pipelines.rtabmap import RTABMap
from pipelines.apriltag import AprilTag
from pipelines.object_tracker import ObjectTracker

NAME = "PurpleRanger"

epilog = """
Modes:
test: Run a NT4 server for debugging
sim: Connect to simulation NT4 server
robot: Connect to robot NT4 server

Pipeline type:
vio: Visual Inertial Odometry pipeline
object: object tracking
apriltag: 2D apriltag tracking
"""

# Initialize variables
pipeline = Pipeline()
app = app = Flask(NAME, static_url_path='', static_folder='web/static', template_folder='web/templates')
video_feed_stop_event = threading.Event()

def generate_video_feed():
    # loop over frames from the output stream
    while not video_feed_stop_event.is_set():
        # wait until the lock is acquired
        with variables.video_lock:
            # check if the output frame is available, otherwise skip
            if variables.video_frame is None: continue
            # encode the frame in JPEG format
            (flag, encoded_image) = cv2.imencode(".jpg", variables.video_frame)
            # ensure the frame was successfully encoded
            if not flag:
                logging.debug("Video frame encoding failure!")
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    return Response(generate_video_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")

def sigint_handler(sig, frame):
    video_feed_stop_event.set()
    nt_instance = ntcore.NetworkTableInstance.getDefault()
    nt_listener_handles = pipeline.exit()
    logging.info("Exiting...")
    if nt_listener_handles is not None:
        for listener_handle in nt_listener_handles:
            nt_instance.removeListener(listener_handle)
    sys.exit(0)

# Main function
if __name__ ==  "__main__":
    # Bind SIGINT handler
    signal.signal(signal.SIGINT, sigint_handler)

    # Init argparse
    parser = argparse.ArgumentParser(
        prog="PurpleRanger",
        description="Publish pose data from DepthAI Basalt VIO",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Add arguments
    parser.add_argument("--mode", required=True, choices=["test", "sim", "robot"], help="select script mode")
    parser.add_argument("--pipeline", required=True, choices=["vio", "object", "apriltag"], help="pipeline type")
    parser.add_argument("--tag-map", help="path to AprilTag map JSON file")
    parser.add_argument("--verbose", action="store_true", help="debug level output")

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    loglevel = logging.INFO
    if args.verbose: loglevel = logging.DEBUG
    logging.basicConfig(
        level=loglevel,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Init NT4
    nt_instance = ntcore.NetworkTableInstance.getDefault()
    table = nt_instance.getTable(NAME)

    # Start NT4 server/client
    match args.mode:
        case 'test':
            nt_instance.startServer()
            logging.info("Test mode, starting NT4 server...")
        case 'sim':
            nt_instance.setServer("localhost")
            logging.info("Simulation mode, connecting to localhost NT4 server...")
        case _:
            logging.info("Robot mode, connecting to robot NT4 server...")

    # Set tag map path
    if args.tag_map:
        spectacle.config["AprilTagMapPath"] = args.tag_map
        logging.info("Using AprilTag map at " + args.tag_map)
    else:
        logging.info("No AprilTag map provided, not using AprilTags!")

    # Start NT4 clients
    nt_instance.startClient4(NAME)
    nt_instance.startDSClient()

    # Clear video feed thread stop event
    video_feed_stop_event.clear()

    # Select pipeline
    match args.pipeline:
        case "object":
            pipeline = ObjectTracker()
        case "vio":
            pipeline = Basalt(table)
        case "apriltag":
            pipeline = AprilTag()
        case _:
            pass

    # Start selected pipeline
    pipeline.start()

    # Start web server
    app.run(host="localhost", port=8080, debug=False, threaded=True, use_reloader=False)

