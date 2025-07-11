import os
import sys
import time
import json
import signal
import logging
import argparse
import threading
import math

import ntcore
from wpiutil import wpistruct
from wpimath.geometry import Pose3d, Translation3d, Rotation3d, Quaternion

import depthai

import variables
from .pipeline import Pipeline


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
            device.setLogLevel(depthai.LogLevel.DEBUG)
            logging.info(device.getDeviceName())

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

            # Define sources and output nodes
            left = p.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_B, sensorFps=fps)
            right = p.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_C, sensorFps=fps)
            imu = p.create(depthai.node.IMU)
            odom = p.create(depthai.node.BasaltVIO)
            slam = p.create(depthai.node.RTABMapSLAM)
            stereo = p.create(depthai.node.StereoDepth)
            params = {
                "RGBD/CreateOccupancyGrid": "true",
                "Grid/3D": "true",
                "Rtabmap/SaveWMState": "true"
            }
            slam.setParams(params)

            # Setup IMU
            imu.enableIMUSensor([depthai.IMUSensor.ACCELEROMETER_RAW, depthai.IMUSensor.GYROSCOPE_RAW], 200)
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)

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
            stereo.syncedLeft.link(odom.left)
            stereo.syncedRight.link(odom.right)
            stereo.depth.link(slam.depth)
            stereo.rectifiedLeft.link(slam.rect)
            imu.out.link(odom.imu)
            odom.transform.link(slam.odom)

            # Create output queues
            passthrough_queue = odom.passthrough.createOutputQueue()
            transform_queue = slam.transform.createOutputQueue()
            imu_queue = imu.out.createOutputQueue()

            # Run pipeline
            p.start()
            logging.info("Basalt VIO initialised")
            logging.info("Config - " + str(self.config))
            while p.isRunning():
                while not self.stop_event.is_set():
                    if not transform_queue.has():
                        time.sleep(WAIT_TIME)
                        continue

                    imgFrame = passthrough_queue.get()
                    transform_message = transform_queue.get()
                    temp_point = transform_message.getTranslation()
                    temp_quaternion = transform_message.getQuaternion()

                    pose = Pose3d(
                        Translation3d(temp_point.x, temp_point.y, temp_point.z),
                        Rotation3d(Quaternion(temp_quaternion.qw, temp_quaternion.qx, temp_quaternion.qy, temp_quaternion.qz))
                    )

                    self.status_publisher.set(True)
                    self.pose_publisher.set(pose)
                    logging.debug(str(pose))

                    frame = imgFrame.getCvFrame()

                    with variables.video_lock:
                        variables.video_frame = frame.copy()

                    time.sleep(WAIT_TIME)

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

