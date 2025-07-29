#!/usr/bin/env python

import signal
import time
import depthai as dai

import numpy as np

from robotpy_apriltag import AprilTagField, AprilTagFieldLayout
from wpimath.geometry import Pose3d, Translation3d, Transform3d, Rotation3d, Quaternion

import variables
from utils.apriltag import AprilTagPoseEstimation, TargetModel, OpenCVHelp

# Create pipeline
with dai.Pipeline() as p:
    device = p.getDefaultDevice()
    calib = device.readCalibration()

    fps = 60
    frame_width = 640
    frame_height = 480

    field_pose_init = False
    field_pose_origin = Pose3d()

    field_layout = AprilTagFieldLayout.loadField(AprilTagField.k2025ReefscapeWelded)

    left_camera_matrix = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, frame_width, frame_height))
    left_dist_coeffs = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
    right_camera_matrix = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, frame_width, frame_height))
    right_dist_coeffs = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))

    # Define sources and outputs
    left = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
    right = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)
    imu = p.create(dai.node.IMU)
    odom = p.create(dai.node.BasaltVIO)
    left_apriltag = p.create(dai.node.AprilTag)
    right_apriltag = p.create(dai.node.AprilTag)

    imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 200)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    left.requestOutput((frame_width, frame_height)).link(odom.left)
    right.requestOutput((frame_width, frame_height)).link(odom.right)
    left.requestOutput((frame_width, frame_height), dai.ImgFrame.Type.GRAY8).link(left_apriltag.inputImage)
    right.requestOutput((frame_width, frame_height), dai.ImgFrame.Type.GRAY8).link(right_apriltag.inputImage)
    imu.out.link(odom.imu)

    transform_queue = odom.transform.createOutputQueue()
    passthrough_output_queue = left_apriltag.passthroughInputImage.createOutputQueue()
    left_output_queue = left_apriltag.out.createOutputQueue()
    right_output_queue = right_apriltag.out.createOutputQueue()

    p.start()
    while p.isRunning():
        left_apriltag_message = left_output_queue.get()
        right_apriltag_message = right_output_queue.get()
        transform_message = transform_queue.get()

        if not field_pose_init:
            left_tags = left_apriltag_message.aprilTags
            right_tags = right_apriltag_message.aprilTags

            passthrough_image: dai.ImgFrame = passthrough_output_queue.get()
            frame = passthrough_image.getCvFrame()

            OpenCVHelp.drawTags(frame, left_tags, (0, 255, 0))

            left_estimate = AprilTagPoseEstimation.estimateCamPosePNP(
                left_camera_matrix,
                left_dist_coeffs,
                left_tags,
                field_layout,
                TargetModel.AprilTag36h11()
            )

            right_estimate = AprilTagPoseEstimation.estimateCamPosePNP(
                right_camera_matrix,
                right_dist_coeffs,
                right_tags,
                field_layout,
                TargetModel.AprilTag36h11()
            )

            field_pose = AprilTagPoseEstimation.mergePoses(left_estimate, right_estimate, variables.baseline)
            if field_pose:
                field_pose_origin = field_pose
                field_pose_init = True
            else: continue

        temp_point = transform_message.getTranslation()
        temp_quaternion = transform_message.getQuaternion()

        basalt_transform = Transform3d(
            Translation3d(temp_point.x, temp_point.y, temp_point.z),
            Rotation3d(Quaternion(temp_quaternion.qw, temp_quaternion.qx, temp_quaternion.qy, temp_quaternion.qz))
        )

        print(str(field_pose_origin.transformBy(basalt_transform)))
        time.sleep(0.01)
