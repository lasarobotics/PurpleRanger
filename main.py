#!/usr/bin/env python

import os
import sys
import time
import json
import signal
import logging
import argparse
import threading

import ntcore
from wpiutil import wpistruct
from wpimath.geometry import Pose3d, Rotation3d, Quaternion

from pipelines.pipeline import Pipeline
from pipelines.basalt import Basalt

global spectacular_thread
global nt_listener_handles

global status_publisher
global pose_publisher
global stop_event

NAME = "PurpleRanger"

epilog = """
Modes:
test: Run a NT4 server for debugging
sim: Connect to simulation NT4 server
robot: Connect to robot NT4 server

Pipeline type:
vio: Visual Inertial Odometry pipeline
inference: AI inference pipeline
"""

pipeline = Pipeline()

def signal_handler(sig, frame):
    logging.info("Exiting...")
    nt_instance = ntcore.NetworkTableInstance.getDefault()
    nt_listener_handles = pipeline.exit()
    for listener_handle in nt_listener_handles:
        nt_instance.removeListener(listener_handle)
    sys.exit(0)

# Main function
if __name__ ==  "__main__":
    # Bind SIGINT handler
    signal.signal(signal.SIGINT, signal_handler)

    # Init argparse
    parser = argparse.ArgumentParser(
        prog="PurpleRanger",
        description="Publish pose data from DepthAI Basalt VIO",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Add arguments
    parser.add_argument("--mode", required=True, choices=["test", "sim", "robot"], help="select script mode")
    parser.add_argument("--pipeline", required=True, choices=["vio", "inference"], help="pipeline type")
    parser.add_argument("--tag-map", help="path to AprilTag map JSON file")
    parser.add_argument("--verbose", action="store_true")

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

    match args.pipeline:
        case "inference":
            pass
        case "vio":
            pipeline = Basalt(table)
        case _:
            pass

    pipeline.start()

    while True:
        time.sleep(0.2)

