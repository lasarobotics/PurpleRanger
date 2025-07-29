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

        ## CHANGED: Particle filter parameters
        self.num_particles = 3000
        self.particles = np.empty(self.num_particles, dtype=object)
        self.weights = np.ones(self.num_particles) / self.num_particles
        # Noise added during the prediction step to simulate VIO drift
        self.motion_noise = [0.001, 0.001, 0.001, 0.002, 0.002, 0.002] # Trans(x,y,z), Rot(r,p,y)

        # VIO transform from the previous frame, needed to calculate delta
        self.last_basalt_transform = None


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


    ## NEW: Initialize particles around the first measurement
    def __initialize_particles(self, initial_pose: Pose3d):
        for i in range(self.num_particles):
            self.particles[i] = initial_pose
        self.field_pose_init = True
        logging.info("Particle filter initialized.")


    ## NEW: Predict step - move particles based on VIO
    def __predict(self, delta_transform: Transform3d):
        for i in range(self.num_particles):
            # Apply VIO's delta transform
            p = self.particles[i].transformBy(delta_transform)

            # Add random noise to simulate drift
            noise_t = Translation3d(np.random.normal(0, self.motion_noise[0]),
                                    np.random.normal(0, self.motion_noise[1]),
                                    np.random.normal(0, self.motion_noise[2]))
            noise_r = Rotation3d(np.random.normal(0, self.motion_noise[3]),
                                 np.random.normal(0, self.motion_noise[4]),
                                 np.random.normal(0, self.motion_noise[5]))

            self.particles[i] = Pose3d(p.translation() + noise_t, p.rotation() + noise_r)


    ## NEW: Update step - weigh particles based on AprilTag measurement
    def __update(self, measurement: Pose3d, measurement_noise_std: list[float]):
        for i in range(self.num_particles):
            # .log() returns a Twist3d object (dx, dy, dz, rx, ry, rz)
            error_twist = self.particles[i].log(measurement)

            ## CHANGED: Manually create a list from the Twist3d components
            error = [error_twist.dx, error_twist.dy, error_twist.dz, error_twist.rx, error_twist.ry, error_twist.rz]

            # Calculate weight using a multivariate Gaussian probability density function.
            # This scores the particle based on its distance from the measurement.
            g = 1.0
            for j in range(6):
                g *= (1.0 / (measurement_noise_std[j] * math.sqrt(2.0 * math.pi))) * \
                      math.exp(-0.5 * (error[j] / measurement_noise_std[j]) ** 2)
            self.weights[i] = g

        # Normalize weights so they sum to 1
        self.weights += 1.e-300 # prevent division by zero
        self.weights /= sum(self.weights)


    ## NEW: Resample step - generate new particle cloud
    def __resample(self):
        new_particles = np.empty(self.num_particles, dtype=object)
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        for i, idx in enumerate(indices):
            new_particles[i] = self.particles[idx]
        self.particles = new_particles
        self.weights.fill(1.0 / self.num_particles)


    ## NEW: Estimate final pose from the particle cloud
    def __estimate_pose(self) -> Pose3d:
        # Calculate the weighted mean of all particles
        mean_t = Translation3d()
        # For rotation, we average the quaternions. This is more stable than averaging Euler angles.
        mean_q = Quaternion(0,0,0,0)

        for i in range(self.num_particles):
            p = self.particles[i]
            w = self.weights[i]
            mean_t += p.translation() * w
            q = p.rotation().getQuaternion()

            # Ensure quaternions are aligned for averaging (handle the q = -q duality)
            if mean_q.dot(q) < 0:
                q = Quaternion(-q.W(), -q.X(), -q.Y(), -q.Z())

            mean_q = Quaternion(mean_q.W() + q.W() * w,
                                mean_q.X() + q.X() * w,
                                mean_q.Y() + q.Y() * w,
                                mean_q.Z() + q.Z() * w)

        mean_q.normalize()
        return Pose3d(mean_t, Rotation3d(mean_q))


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

            # Run pipeline
            p.start()
            logging.info("Basalt Simple initialised")
            logging.info("Config - " + str(self.config))
            while p.isRunning() and not self.stop_event.is_set():
                image = image_queue.get()
                transform_message = transform_queue.get()
                left_tag_message = left_tag_queue.get()
                right_tag_message = right_tag_queue.get()
                assert isinstance(image, depthai.ImgFrame), "Expected ImgFrame"
                assert isinstance(transform_message, depthai.TransformData), "Expected TransformData"
                assert isinstance(left_tag_message, depthai.AprilTags), "Expected AprilTags"
                assert isinstance(right_tag_message, depthai.AprilTags), "Expected AprilTags"

                # VIO measurement
                temp_point = transform_message.getTranslation()
                temp_quaternion = transform_message.getQuaternion()
                current_basalt_transform = Transform3d(
                    Translation3d(temp_point.x, temp_point.y, temp_point.z),
                    Rotation3d(Quaternion(temp_quaternion.qw, temp_quaternion.qx, temp_quaternion.qy, temp_quaternion.qz))
                )

                if self.last_basalt_transform is None:
                    self.last_basalt_transform = current_basalt_transform

                # Get delta since last loop
                delta_transform = self.last_basalt_transform.inverse() + current_basalt_transform
                self.last_basalt_transform = current_basalt_transform

                # Particle Filter: Predict Step
                if self.field_pose_init:
                    self.__predict(delta_transform)

                # AprilTag measurement
                left_estimate = AprilTagPoseEstimation.estimateCamPosePNP(left_camera_matrix, left_dist_coeffs, left_tag_message.aprilTags, field_layout, TargetModel.AprilTag36h11())
                right_estimate = AprilTagPoseEstimation.estimateCamPosePNP(right_camera_matrix, right_dist_coeffs, right_tag_message.aprilTags, field_layout, TargetModel.AprilTag36h11())
                field_pose_measurement, measurement_noise_std = AprilTagPoseEstimation.mergePoses(left_estimate, right_estimate, field_layout, variables.baseline, Perspective.LEFT)

                if not self.field_pose_init and field_pose_measurement:
                    # First measurement, initialize the filter
                    self.__initialize_particles(field_pose_measurement)

                elif field_pose_measurement:
                    # Particle Filter: Update and Resample Steps
                    self.__update(field_pose_measurement, measurement_noise_std)
                    self.__resample()

                if not self.field_pose_init:
                    # Do nothing until we see a tag
                    continue

                # Get final pose estimate
                final_pose = self.__estimate_pose()

                self.status_publisher.set(True)
                self.pose_publisher.set(final_pose)
                logging.debug(final_pose)

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