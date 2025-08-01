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
from wpimath.geometry import Pose3d, Translation3d, Rotation3d, Quaternion

import depthai

import variables
from .pipeline import Pipeline
from utils.apriltag import TargetModel, OpenCVHelp
from utils.nodes import LandmarkEstimator


class Basalt(Pipeline):
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
            marker_priors = ""
            for tag in field_layout.getTags():
                marker_priors += " ".join([
                    str(tag.ID),
                    str(tag.pose.translation().X()),
                    str(tag.pose.translation().Y()),
                    str(tag.pose.translation().Z()),
                    str(tag.pose.rotation().X()),
                    str(tag.pose.rotation().Y()),
                    str(tag.pose.rotation().Z()),
                    "|"
                ])

            if "OAK-D-PRO" in device.getDeviceName():
                device.setIrLaserDotProjectorIntensity(self.config["DotProjectorIntensity"])
                device.setIrFloodLightIntensity(self.config["IRFloodlightIntensity"])

            fps = 30
            frame_width = 640
            frame_height = 480

            if "x86" in platform.machine():
                fps = 60

            if "OAK-D-LITE" in device.getDeviceName():
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
            stereo = p.create(depthai.node.StereoDepth)
            feature_tracker = p.create(depthai.node.FeatureTracker)
            params = {
                "RGBD/CreateOccupancyGrid": "true",
                "Grid/3D": "true",
                "Rtabmap/SaveWMState": "true",
                "RGBD/MarkerDetection": "true",
                "Optimizer/Strategy": "2",
                "Optimizer/Iterations": "50",
                "Optimizer/PriorsIgnored": "false",
                "Optimizer/GravitySigma": "0.3",
                "Marker/VarianceOrientationIgnored": "true",
                "Marker/Priors": marker_priors,
            }
            slam.setParams(params)
            slam.setUseFeatures(True)
            slam.setUseLandmarks(True)
            landmark_estimator = LandmarkEstimator()

            # Setup IMU
            imu.enableIMUSensor([depthai.IMUSensor.ACCELEROMETER_RAW, depthai.IMUSensor.GYROSCOPE_RAW], 200)
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)

            # Setup landmark estimator
            landmark_estimator.setCameraIntrinsics(left_camera_matrix, right_camera_matrix)
            landmark_estimator.setDistortionCoefficients(left_dist_coeffs, right_dist_coeffs)
            landmark_estimator.setTargetModel(TargetModel.AprilTag36h11())
            landmark_estimator.setAprilTagFieldLayout(field_layout)

            # Setup stereo
            stereo.setExtendedDisparity(False)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(False)
            stereo.setRectifyEdgeFillColor(0)
            stereo.enableDistortionCorrection(True)
            stereo.initialConfig.setLeftRightCheckThreshold(10)
            stereo.setDepthAlign(depthai.CameraBoardSocket.CAM_B)

            # Setup feature tracker
            feature_tracker.setHardwareResources(2, 2)
            feature_tracker.initialConfig.setCornerDetector(depthai.FeatureTrackerConfig.CornerDetector.Type.HARRIS)
            feature_tracker.initialConfig.setNumTargetFeatures(3000)
            feature_tracker.initialConfig.setMotionEstimator(False)
            feature_tracker.initialConfig.FeatureMaintainer.minimumDistanceBetweenFeatures = 9

            # Link nodes
            left.requestOutput((frame_width, frame_height)).link(stereo.left)
            right.requestOutput((frame_width, frame_height)).link(stereo.right)
            left.requestOutput((frame_width, frame_height), depthai.ImgFrame.Type.GRAY8).link(left_apriltag.inputImage)
            right.requestOutput((frame_width, frame_height), depthai.ImgFrame.Type.GRAY8).link(right_apriltag.inputImage)
            left_apriltag.out.link(landmark_estimator.leftTags)
            right_apriltag.out.link(landmark_estimator.rightTags)
            stereo.syncedLeft.link(odom.left)
            stereo.syncedRight.link(odom.right)
            stereo.depth.link(slam.depth)
            stereo.rectifiedLeft.link(slam.rect)
            stereo.rectifiedLeft.link(feature_tracker.inputImage)
            feature_tracker.outputFeatures.link(slam.features)
            landmark_estimator.landmarks.link(slam.landmarks)
            imu.out.link(odom.imu)
            imu.out.link(slam.imu)
            odom.transform.link(slam.odom)

            # Create output queues
            image_queue = odom.passthrough.createOutputQueue()
            transform_queue = slam.transform.createOutputQueue()
            landmark_queue = landmark_estimator.landmarks.createOutputQueue()
            left_tag_queue = left_apriltag.out.createOutputQueue()

            # Run pipeline
            p.start()
            logging.info("Basalt VIO initialised")
            logging.info("Config - " + str(self.config))
            while p.isRunning() and not self.stop_event.is_set():
                image = image_queue.get()
                transform_message = transform_queue.get()
                landmark_message = landmark_queue.get()
                left_tag_message = left_tag_queue.get()
                assert isinstance(image, depthai.ImgFrame), "Expected ImgFrame"
                assert isinstance(transform_message, depthai.TransformData), "Expected TransformData"
                assert isinstance(landmark_message, depthai.Landmarks), "Expected Landmarks"


                temp_point = transform_message.getTranslation()
                temp_quaternion = transform_message.getQuaternion()

                pose = Pose3d(
                    Translation3d(temp_point.x, temp_point.y, temp_point.z),
                    Rotation3d(Quaternion(temp_quaternion.qw, temp_quaternion.qx, temp_quaternion.qy, temp_quaternion.qz))
                )

                self.status_publisher.set(True)
                self.pose_publisher.set(pose)
                logging.debug(str(pose))

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

