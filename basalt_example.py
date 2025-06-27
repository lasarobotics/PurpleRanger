#!/usr/bin/env python

import signal
import time
import depthai as dai

from wpimath.geometry import Pose3d, Translation3d, Rotation3d, Quaternion

# Create pipeline
with dai.Pipeline() as p:
    fps = 60
    width = 640
    height = 400
    # Define sources and outputs
    left = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
    right = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)
    imu = p.create(dai.node.IMU)
    odom = p.create(dai.node.BasaltVIO)

    imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 200)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    # Linking
    left.requestOutput((width, height)).link(odom.left)
    right.requestOutput((width, height)).link(odom.right)
    imu.out.link(odom.imu)
    transform_queue = odom.transform.createOutputQueue()

    p.start()
    while p.isRunning():
        transform_message = transform_queue.get()
        temp_point = transform_message.getTranslation()
        temp_quaternion = transform_message.getQuaternion()

        pose = Pose3d(
            Translation3d(temp_point.x, temp_point.y, temp_point.z),
            Rotation3d(Quaternion(temp_quaternion.qw, temp_quaternion.qx, temp_quaternion.qy, temp_quaternion.qz))
        )

        #print(str(pose))
        time.sleep(0.01)
