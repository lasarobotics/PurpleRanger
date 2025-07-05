#!/usr/bin/env python

import time
import logging
import threading

import numpy as np

import cv2

import depthai

import ntcore
from robotpy_apriltag import AprilTagField, AprilTagFieldLayout
from wpimath.geometry import Pose3d, Twist3d, Transform3d, Translation3d, Rotation3d, Quaternion

import variables
from .pipeline import Pipeline
from utils.apriltag import AprilTagPoseEstimation, TargetModel, OpenCVHelp

EPSILON = 1e-6
BASELINE = 0.075

class AprilTag3D(Pipeline):
    def __init__(self, table: ntcore.NetworkTable):
        self.stop_event = threading.Event()
        self.__nt_init(table)
        self.color = (255, 0, 255)
        self.text_thickness = 3
        self.start_time = time.monotonic()
        self.fps = 120

    def __nt_init(self, table: ntcore.NetworkTable):
        nt_instance = table.getInstance()

        # Create NT4 output publishers
        self.status_publisher = table.getBooleanTopic("Status").publish(ntcore.PubSubOptions(keepDuplicates=True, sendAll=True))
        self.pose_publisher = table.getStructTopic("Pose", Pose3d).publish(ntcore.PubSubOptions(keepDuplicates=True, sendAll=True))

    def __session(self):
        with depthai.Pipeline() as p:
            device = p.getDefaultDevice()
            calib = device.readCalibration()
            frame_width = 1280
            frame_height = 800
            if "OAK-D-LITE" in device.getDeviceName():
                frame_width = 640
                frame_height = 480
            left: depthai.node.Camera = p.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_B)
            right = p.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_C)
            left_apriltag_node = p.create(depthai.node.AprilTag)
            right_apriltag_node = p.create(depthai.node.AprilTag)
            left.requestOutput((frame_width, frame_height), depthai.ImgFrame.Type.BGR888p).link(left_apriltag_node.inputImage)
            right.requestOutput((frame_width, frame_height), depthai.ImgFrame.Type.BGR888p).link(right_apriltag_node.inputImage)
            passthrough_output_queue = left_apriltag_node.passthroughInputImage.createOutputQueue()
            left_output_queue = left_apriltag_node.out.createOutputQueue()
            right_output_queue = right_apriltag_node.out.createOutputQueue()

            field_layout = AprilTagFieldLayout.loadField(AprilTagField.k2025ReefscapeWelded)

            p.start()
            logging.info("AprilTag tracker initialised")
            while p.isRunning():
                while not self.stop_event.is_set():
                    left_apriltag_message = left_output_queue.get()
                    right_apriltag_message = right_output_queue.get()

                    left_tags = left_apriltag_message.aprilTags
                    right_tags = right_apriltag_message.aprilTags

                    passthrough_image: depthai.ImgFrame = passthrough_output_queue.get()
                    frame = passthrough_image.getCvFrame()

                    OpenCVHelp.drawTags(frame, left_tags, self.color)

                    left_estimate = AprilTagPoseEstimation.estimateCamPosePNP(
                        np.array(calib.getCameraIntrinsics(depthai.CameraBoardSocket.CAM_B, frame_width, frame_height)),
                        np.array(calib.getDistortionCoefficients(depthai.CameraBoardSocket.CAM_B)),
                        left_tags,
                        field_layout,
                        TargetModel.AprilTag36h11()
                    )

                    right_estimate = AprilTagPoseEstimation.estimateCamPosePNP(
                        np.array(calib.getCameraIntrinsics(depthai.CameraBoardSocket.CAM_C, frame_width, frame_height)),
                        np.array(calib.getDistortionCoefficients(depthai.CameraBoardSocket.CAM_C)),
                        right_tags,
                        field_layout,
                        TargetModel.AprilTag36h11()
                    )

                    pose = AprilTagPoseEstimation.mergePoses(left_estimate, right_estimate, variables.baseline)
                    self.status_publisher.set(pose is not None)
                    if pose:
                        self.pose_publisher.set(pose)
                        logging.debug(str(pose))

                    # Copy frame for output
                    with variables.video_lock:
                        variables.video_frame = frame.copy()

                p.stop()
                logging.info("AprilTag tracker stopped")


    def start(self):
        self.stop_event.clear()
        self.apriltag_thread = threading.Thread(target=self.__session)
        self.apriltag_thread.start()


    def stop(self):
        self.stop_event.set()
        self.apriltag_thread.join()
        time.sleep(1)


    def exit(self):
        self.stop()
