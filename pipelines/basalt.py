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
from utils.apriltag import TargetModel
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
            device.setLogLevel(depthai.LogLevel.DEBUG)
            logging.info(device.getDeviceName())

            device.setLogLevel(depthai.LogLevel.DEBUG)
            device.setLogOutputLevel(depthai.LogLevel.DEBUG)

            field_layout = AprilTagFieldLayout.loadField(AprilTagField.k2025ReefscapeWelded)

            if "OAK-D-PRO" in device.getDeviceName():
                device.setIrLaserDotProjectorIntensity(self.config["DotProjectorIntensity"])
                device.setIrFloodLightIntensity(self.config["IRFloodlightIntensity"])

            fps = 120
            frame_width = 1280
            frame_height = 800

            if "OAK-D-LITE" in device.getDeviceName():
                fps = 90
                frame_width = 640
                frame_height = 480

            # Get intrisics matrix and distortion coefficients
            camera_matrix = np.array(calib.getCameraIntrinsics(depthai.CameraBoardSocket.CAM_B, frame_width, frame_height))
            dist_coeffs = np.array(calib.getDistortionCoefficients(depthai.CameraBoardSocket.CAM_B))

            # Define sources and output nodes
            left = p.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_B, sensorFps=fps)
            right = p.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_C, sensorFps=fps)
            apriltag = p.create(depthai.node.AprilTag)
            imu = p.create(depthai.node.IMU)
            odom = p.create(depthai.node.BasaltVIO)
            slam = p.create(depthai.node.RTABMapSLAM)
            stereo = p.create(depthai.node.StereoDepth)
            params = {
                "RGBD/CreateOccupancyGrid": "true",
                "Grid/3D": "true",
                "Rtabmap/SaveWMState": "true",
                "RGBD/MarkerDetection": "true",
                "Optimizer/PriorsIgnored": "false",
                "Marker/Priors": "10 12.227305999999999 4.0259 0.308102 0 0 3.14159"
            }
            slam.setParams(params)
            slam.setUseLandmarks(True)
            landmark_estimator = LandmarkEstimator()

            # Setup IMU
            imu.enableIMUSensor([depthai.IMUSensor.ACCELEROMETER_RAW, depthai.IMUSensor.GYROSCOPE_RAW], 200)
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)

            # Setup landmark estimator
            landmark_estimator.setCameraIntrinsics(camera_matrix)
            landmark_estimator.setDistortionCoefficients(dist_coeffs)
            landmark_estimator.setTargetModel(TargetModel.AprilTag36h11())
            landmark_estimator.setAprilTagFieldLayout(field_layout)

            # Setup stereo
            stereo.setExtendedDisparity(False)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            stereo.setRectifyEdgeFillColor(0)
            stereo.enableDistortionCorrection(True)
            stereo.initialConfig.setLeftRightCheckThreshold(10)
            stereo.setDepthAlign(depthai.CameraBoardSocket.CAM_B)

            # Link nodes
            left.requestOutput((frame_width, frame_height)).link(stereo.left)
            right.requestOutput((frame_width, frame_height)).link(stereo.right)
            left.requestOutput((frame_width, frame_height), depthai.ImgFrame.Type.GRAY8).link(apriltag.inputImage)
            apriltag.out.link(landmark_estimator.tags)
            stereo.syncedLeft.link(odom.left)
            stereo.syncedRight.link(odom.right)
            stereo.depth.link(slam.depth)
            stereo.rectifiedLeft.link(slam.rect)
            landmark_estimator.landmarks.link(slam.landmarks)
            imu.out.link(odom.imu)
            odom.transform.link(slam.odom)

            # Create output queues
            image_queue = odom.passthrough.createOutputQueue()
            transform_queue = slam.transform.createOutputQueue()
            landmark_queue = landmark_estimator.landmarks.createOutputQueue()
            passthrough_queue = slam.passthroughFeatures.createOutputQueue()

            # Run pipeline
            p.start()
            logging.info("Basalt VIO initialised")
            logging.info("Config - " + str(self.config))
            while p.isRunning() and not self.stop_event.is_set():
                image = image_queue.get()
                transform_message = transform_queue.get()
                landmark_message = landmark_queue.get()
                assert isinstance(image, depthai.ImgFrame), "Expected ImgFrame"
                assert isinstance(transform_message, depthai.TransformData), "Expected TransformData"
                assert isinstance(landmark_message, depthai.Landmarks), "Expected Landmarks"

                logging.debug(str(landmark_message))

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

