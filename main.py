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
from wpimath.geometry import Pose3d, Translation3d, Rotation3d, Quaternion

import depthai

# Create pipeline
with depthai.Pipeline() as p:
    fps = 120
    width = 640
    height = 480

    depthai.CameraModel

    # Define sources and output nodes
    left = p.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_B, sensorFps=fps)
    right = p.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_C, sensorFps=fps)
    imu = p.create(depthai.node.IMU)
    odometry = p.create(depthai.node.BasaltVIO)

    imu.enableIMUSensor([depthai.IMUSensor.ACCELEROMETER_RAW, depthai.IMUSensor.GYROSCOPE_RAW], 200)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    # Link nodes
    left.requestOutput((width, height)).link(odometry.left)
    right.requestOutput((width, height)).link(odometry.right)
    imu.out.link(odometry.imu)

    # Create output queue
    output = odometry.transform.createOutputQueue()

    # Run pipeline
    p.start()
    while p.isRunning():
        transform_message = output.get()
        temp_point = transform_message.getTranslation()
        temp_quaternion = transform_message.getQuaternion()

        pose = Pose3d(
            Translation3d(temp_point.x, temp_point.y, temp_point.z),
            Rotation3d(Quaternion(temp_quaternion.qw, temp_quaternion.qx, temp_quaternion.qy, temp_quaternion.qz))
        )

        print(str(pose))

        time.sleep(0.01)

