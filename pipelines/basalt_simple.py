import os
import sys
import time
import json
import signal
import logging
import argparse
import threading
import math

import numpy as np

import ntcore
from wpiutil import wpistruct
from robotpy_apriltag import AprilTagField, AprilTagFieldLayout
from wpimath.geometry import Pose3d, Translation3d, Rotation3d, Transform3d, Quaternion

import depthai

import variables
from .pipeline import Pipeline
from utils.apriltag import AprilTagPoseEstimation, TargetModel, OpenCVHelp, Perspective

CORRECTION_THRESHOLD = Transform3d(Translation3d(0.03, 0.03, 0.03), Rotation3d(math.radians(5.0), math.radians(5.0), math.radians(5.0)))


class BasaltSimple(Pipeline):
    def __init__(self, table: ntcore.NetworkTable):
        self.config = {
            "AutoExposure": True,
            "DotProjectorIntensity": 0.0,
            "IRFloodlightIntensity": 0.0,
            "AprilTagMapPath": "",
        }
        self.stop_event = threading.Event()
        self.__nt_init(table)
        self.stop_event.clear()

        self.field_pose_init = False
        self.field_pose_origin = Pose3d()


    def __nt_init(self, table: ntcore.NetworkTable):
        nt_instance = table.getInstance()

        logging.info("Initializing network table...")
        # Create NT4 output publishers
        self.status_publisher = table.getBooleanTopic("Status").publish(ntcore.PubSubOptions(keepDuplicates=True, sendAll=True))
        self.pose_publisher = table.getStructTopic("Pose", Pose3d).publish(ntcore.PubSubOptions(keepDuplicates=True, sendAll=True))

        # Create NT4 config entries
        topics = list(self.config.keys())
        auto_exposure_entry = table.getBooleanTopic(topics[0]).getEntry(self.config[topics[0]])
        auto_exposure_entry.setDefault(self.config[topics[0]])
        time.sleep(1)
        dot_projector_intensity_entry = table.getDoubleTopic(topics[1]).getEntry(self.config[topics[1]])
        dot_projector_intensity_entry.setDefault(self.config[topics[1]])
        time.sleep(1)
        ir_floodlight_intensity_entry = table.getDoubleTopic(topics[2]).getEntry(self.config[topics[2]])
        ir_floodlight_intensity_entry.setDefault(self.config[topics[2]])
        time.sleep(1)
        apriltag_map_path_entry = table.getStringTopic(topics[3]).getEntry(self.config[topics[3]])
        apriltag_map_path_entry.setDefault(self.config[topics[3]])
        time.sleep(1)

        # Put entries in a list
        self.config_entries = []
        self.config_entries.append(auto_exposure_entry)
        self.config_entries.append(dot_projector_intensity_entry)
        self.config_entries.append(ir_floodlight_intensity_entry)
        self.config_entries.append(apriltag_map_path_entry)

    def __session(self):
        # Create pipeline
        with depthai.Pipeline() as p:
            self.status_publisher.set(False)
            device = p.getDefaultDevice()
            calib = device.readCalibration()
            logging.info(device.getDeviceName())

            if variables.trace:
                device.setLogLevel(depthai.LogLevel.DEBUG)
                device.setLogOutputLevel(depthai.LogLevel.DEBUG)

            field_layout = AprilTagFieldLayout.loadField(AprilTagField.k2025ReefscapeWelded)

            if "OAK-D-PRO" in device.getDeviceName():
                device.setIrLaserDotProjectorIntensity(self.config["DotProjectorIntensity"])
                device.setIrFloodLightIntensity(self.config["IRFloodlightIntensity"])

            fps = 60
            frame_width = 640
            frame_height = 480

            if "OAK-D-LITE" in device.getDeviceName():
                fps = 90
                frame_width = 640
                frame_height = 480

            # Get intrisics matrix and distortion coefficients
            left_camera_matrix = np.array(calib.getCameraIntrinsics(depthai.CameraBoardSocket.CAM_B, frame_width, frame_height))
            left_dist_coeffs = np.array(calib.getDistortionCoefficients(depthai.CameraBoardSocket.CAM_B))
            right_camera_matrix = np.array(calib.getCameraIntrinsics(depthai.CameraBoardSocket.CAM_C, frame_width, frame_height))
            right_dist_coeffs = np.array(calib.getDistortionCoefficients(depthai.CameraBoardSocket.CAM_C))

            # Define sources and output nodes
            left = p.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_B, sensorFps=fps)
            right = p.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_C, sensorFps=fps)
            left_apriltag = p.create(depthai.node.AprilTag)
            right_apriltag = p.create(depthai.node.AprilTag)
            imu = p.create(depthai.node.IMU)
            odom = p.create(depthai.node.BasaltVIO)
            slam = p.create(depthai.node.RTABMapSLAM)

            # Setup IMU
            imu.enableIMUSensor([depthai.IMUSensor.ACCELEROMETER_RAW, depthai.IMUSensor.GYROSCOPE_RAW], 200)
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)

            # Link nodes
            left.requestOutput((frame_width, frame_height)).link(odom.left)
            right.requestOutput((frame_width, frame_height)).link(odom.right)
            left.requestOutput((frame_width, frame_height), depthai.ImgFrame.Type.GRAY8).link(left_apriltag.inputImage)
            right.requestOutput((frame_width, frame_height), depthai.ImgFrame.Type.GRAY8).link(right_apriltag.inputImage)
            imu.out.link(odom.imu)

            # Create output queues
            image_queue = odom.passthrough.createOutputQueue()
            transform_queue = odom.transform.createOutputQueue()
            left_tag_queue = left_apriltag.out.createOutputQueue()
            right_tag_queue = right_apriltag.out.createOutputQueue()
            imu_queue = imu.out.createOutputQueue(maxSize=50, blocking=False)

            # Run pipeline
            p.start()
            logging.info("Basalt Simple initialised")
            logging.info("Config - " + str(self.config))
            while p.isRunning() and not self.stop_event.is_set():
                image = image_queue.get()
                transform_message = transform_queue.get()
                left_tag_message = left_tag_queue.get()
                right_tag_message = right_tag_queue.get()
                imu_message = imu_queue.get()
                assert isinstance(image, depthai.ImgFrame), "Expected ImgFrame"
                assert isinstance(transform_message, depthai.TransformData), "Expected TransformData"

                for imu_packet in imu_message.packets:
                    pass

                left_estimate = AprilTagPoseEstimation.estimateCamPosePNP(
                    left_camera_matrix,
                    left_dist_coeffs,
                    left_tag_message.aprilTags,
                    field_layout,
                    TargetModel.AprilTag36h11()
                )

                right_estimate = AprilTagPoseEstimation.estimateCamPosePNP(
                    right_camera_matrix,
                    right_dist_coeffs,
                    right_tag_message.aprilTags,
                    field_layout,
                    TargetModel.AprilTag36h11()
                )

                field_pose = AprilTagPoseEstimation.mergePoses(left_estimate, right_estimate, variables.baseline, Perspective.LEFT)

                temp_point = transform_message.getTranslation()
                temp_quaternion = transform_message.getQuaternion()

                current_basalt_transform = Transform3d(
                    Translation3d(temp_point.x, temp_point.y, temp_point.z),
                    Rotation3d(Quaternion(temp_quaternion.qw, temp_quaternion.qx, temp_quaternion.qy, temp_quaternion.qz))
                )

                if not self.field_pose_init and field_pose:
                    self.field_pose_origin = field_pose
                    self.basalt_transform = current_basalt_transform
                    self.field_pose_init = True

                ## CHANGED: The entire correction logic is updated.
                # Calculate the change in VIO position since the last origin was set.
                delta_transform = self.basalt_transform.inverse() + current_basalt_transform
                # Apply that change to the field-relative origin pose.
                pose = self.field_pose_origin.transformBy(delta_transform)

                error = Transform3d()
                if field_pose:
                    error = pose - field_pose
                    if abs(error.translation().X()) > CORRECTION_THRESHOLD.translation().X() \
                        or abs(error.translation().Y()) > CORRECTION_THRESHOLD.translation().Y() \
                        or abs(error.translation().Z()) > CORRECTION_THRESHOLD.translation().Z() \
                        or abs(error.rotation().X()) > CORRECTION_THRESHOLD.rotation().X() \
                        or abs(error.rotation().Y()) > CORRECTION_THRESHOLD.rotation().Y() \
                        or abs(error.rotation().Z()) > CORRECTION_THRESHOLD.rotation().Z():
                        # If the error is too large, reset the origin to the new AprilTag pose.
                        self.field_pose_origin = field_pose
                        # Also reset the VIO origin to the current VIO transform.
                        self.basalt_transform = current_basalt_transform
                        # The final pose for this frame is the new origin.
                        pose = self.field_pose_origin
                        # pass

                self.status_publisher.set(True)
                self.pose_publisher.set(pose)
                logging.debug(str(error))

                frame = image.getCvFrame()
                OpenCVHelp.drawTags(frame, left_tag_message.aprilTags, (0, 0, 0))
                with variables.video_lock:
                    variables.video_frame = frame.copy()

            p.stop()
            logging.info("Basalt VIO stopped")


    def start(self):
        self.stop_event.clear()
        self.basalt_thread = threading.Thread(target=self.__session)
        self.basalt_thread.start()


    def stop(self):
        self.stop_event.set()
        self.basalt_thread.join()
        time.sleep(1)


    def get_config_entries(self) -> list[ntcore.NetworkTableEntry]:
        return self.config_entries


    def exit(self):
        self.stop()

