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

from pipelines.pipeline import Pipeline

WAIT_TIME = 0.005

# Config variables
config = {
    "AutoExposure": True,
    "DotProjectorIntensity": 0.9,
    "IRFloodlightIntensity": 0.0,
    "AprilTagMapPath": "",
}


class Basalt(Pipeline):
    def __init__(self, table: ntcore.NetworkTable):
        self.stop_event = threading.Event()
        self.__nt_init(ntcore.NetworkTableInstance.getDefault(), table)
        self.stop_event.clear()


    def __on_config_change(event: ntcore.Event):
        """NT4 config change callback

        Stops Basalt VIO session, updates config, and restarts session

        Args:
            event (ntcore.Event): NT4 event
        """

        stop()
        config = super()._update_config(config, event)
        start()


    def __nt_init(self, nt_instance: ntcore.NetworkTableInstance, table: ntcore.NetworkTable):
        # Create NT4 output publishers
        self.status_publisher = table.getBooleanTopic("Status").publish(ntcore.PubSubOptions(keepDuplicates=True, sendAll=True))
        self.pose_publisher = table.getStructTopic("Pose", Pose3d).publish(ntcore.PubSubOptions(keepDuplicates=True, sendAll=True))

        # Create NT4 config entries
        topics = list(config.keys())
        auto_exposure_entry = table.getBooleanTopic(topics[0]).getEntry(config[topics[0]])
        auto_exposure_entry.setDefault(config[topics[0]])
        time.sleep(1)
        dot_projector_intensity_entry = table.getDoubleTopic(topics[1]).getEntry(config[topics[1]])
        dot_projector_intensity_entry.setDefault(config[topics[1]])
        time.sleep(1)
        ir_floodlight_intensity_entry = table.getDoubleTopic(topics[2]).getEntry(config[topics[2]])
        ir_floodlight_intensity_entry.setDefault(config[topics[2]])
        time.sleep(1)
        apriltag_map_path_entry = table.getStringTopic(topics[3]).getEntry(config[topics[3]])
        apriltag_map_path_entry.setDefault(config[topics[3]])
        time.sleep(1)

        # Bind listener callback to subscribers
        self.nt_listener_handles = []
        self.nt_listener_handles.append(nt_instance.addListener(auto_exposure_entry, ntcore.EventFlags.kValueAll, self.__on_config_change))
        self.nt_listener_handles.append(nt_instance.addListener(dot_projector_intensity_entry, ntcore.EventFlags.kValueAll, self.__on_config_change))
        self.nt_listener_handles.append(nt_instance.addListener(ir_floodlight_intensity_entry, ntcore.EventFlags.kValueAll, self.__on_config_change))
        self.nt_listener_handles.append(nt_instance.addListener(apriltag_map_path_entry, ntcore.EventFlags.kValueAll, self.__on_config_change))


    def __basalt_session(self):
        # Create pipeline
        with depthai.Pipeline() as basalt_pipeline:
            self.status_publisher.set(False)
            device = basalt_pipeline.getDefaultDevice()
            logging.info(device.getDeviceName())

            if "OAK-D-PRO" in device.getDeviceName():
                device.setIrLaserDotProjectorIntensity(config["DotProjectorIntensity"])
                device.setIrFloodLightIntensity(config["IRFloodlightIntensity"])

            device.setTimesync(True)

            fps = 120
            width = 640
            height = 480

            # Define sources and output nodes
            left = basalt_pipeline.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_B, sensorFps=fps)
            right = basalt_pipeline.create(depthai.node.Camera).build(depthai.CameraBoardSocket.CAM_C, sensorFps=fps)
            imu = basalt_pipeline.create(depthai.node.IMU)
            odometry = basalt_pipeline.create(depthai.node.BasaltVIO)

            imu.enableIMUSensor([depthai.IMUSensor.ACCELEROMETER, depthai.IMUSensor.GYROSCOPE_RAW], 200)
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)

            # Link nodes
            left.requestOutput((width, height)).link(odometry.left)
            right.requestOutput((width, height)).link(odometry.right)
            imu.out.link(odometry.imu)

            # Create output queue
            output = odometry.transform.createOutputQueue()

            # Run pipeline
            basalt_pipeline.start()
            logging.info("Basalt VIO initialised")
            while basalt_pipeline.isRunning():
                while not self.stop_event.is_set():
                    if not output.has():
                        time.sleep(WAIT_TIME)
                        continue

                    transform_message = output.get()
                    temp_point = transform_message.getTranslation()
                    temp_quaternion = transform_message.getQuaternion()

                    pose = Pose3d(
                        Translation3d(temp_point.x, temp_point.y, temp_point.z),
                        Rotation3d(Quaternion(temp_quaternion.qw, temp_quaternion.qx, temp_quaternion.qy, temp_quaternion.qz))
                    )

                    self.status_publisher.set(True)
                    self.pose_publisher.set(pose)
                    logging.debug(str(pose))

                    time.sleep(WAIT_TIME)

                basalt_pipeline.stop()
                logging.info("Basalt VIO stopped")


    def start(self):
        """Start Basalt VIO session
        """

        self.stop_event.clear()
        self.basalt_thread = threading.Thread(target=self.__basalt_session)
        self.basalt_thread.start()


    def stop(self):
        """Stop Basalt VIO session
        """

        self.stop_event.set()
        self.basalt_thread.join()
        time.sleep(1)

    def exit(self):
        self.stop()
        return self.nt_listener_handles

