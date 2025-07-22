import logging
import marshal

import depthai

import numpy as np

from robotpy_apriltag import AprilTagFieldLayout
from wpimath.geometry import Transform3d, Pose3d, Translation3d

import variables
from utils.apriltag import OpenCVHelp, TargetModel, TagCorner, AprilTagPoseEstimation, Perspective

translation_origin = Translation3d()
rotation_covariance = 0.05

class LandmarkEstimator(depthai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.leftTags = self.createInput()
        self.rightTags = self.createInput()
        self.landmarks = self.createOutput()
        self.tagBlacklist = []


    def run(self):
        while self.isRunning():
            left_buffer = self.leftTags.get() # Get a buffer from the input queue
            right_buffer = self.rightTags.get()
            output_buffer = depthai.Landmarks()
            visible_landmarks = []

            left_estimate = AprilTagPoseEstimation.estimateCamPosePNP(self.leftCameraMatrix, self.leftDistCoeffs, left_buffer.aprilTags, self.fieldLayout, self.targetModel)
            right_estimate = AprilTagPoseEstimation.estimateCamPosePNP(self.rightCameraMatrix, self.rightDistCoeffs, right_buffer.aprilTags, self.fieldLayout, self.targetModel)
            if left_estimate and right_estimate:
                pose = AprilTagPoseEstimation.mergePoses(left_estimate, right_estimate, variables.baseline, Perspective.LEFT)
                common_tags = list(set(left_estimate.fiducialIDsUsed) & set(right_estimate.fiducialIDsUsed))
                for tagID in common_tags:
                    camToTag = Transform3d(pose, self.fieldLayout.getTagPose(tagID))
                    distance = camToTag.translation().distance(translation_origin)
                    covariance_matrix = [[0] * 6 for i in range(6)]
                    translation_covariance = 0.01 * (distance ** 2) / len(common_tags)
                    covariance_matrix[0][0] = translation_covariance
                    covariance_matrix[1][1] = translation_covariance
                    covariance_matrix[2][2] = translation_covariance
                    covariance_matrix[3][3] = rotation_covariance
                    covariance_matrix[4][4] = rotation_covariance
                    covariance_matrix[5][5] = rotation_covariance
                    logging.debug(str(covariance_matrix))
                    landmark = depthai.Landmark()
                    landmark.id = tagID
                    landmark.size = self.size
                    landmark.translation.x = camToTag.translation().X()
                    landmark.translation.y = camToTag.translation().Y()
                    landmark.translation.z = camToTag.translation().Z()
                    landmark.quaternion.qx = camToTag.rotation().getQuaternion().X()
                    landmark.quaternion.qy = camToTag.rotation().getQuaternion().Y()
                    landmark.quaternion.qz = camToTag.rotation().getQuaternion().Z()
                    landmark.quaternion.qw = camToTag.rotation().getQuaternion().W()
                    landmark.covariance = covariance_matrix

                    visible_landmarks.append(landmark)

            output_buffer.setTimestamp(left_buffer.getTimestamp())
            output_buffer.setTimestampDevice(left_buffer.getTimestampDevice())
            output_buffer.landmarks = visible_landmarks
            self.landmarks.send(output_buffer)


    def setCameraIntrinsics(self, leftCameraMatrix: np.ndarray, rightCameraMatrix: np.ndarray):
        """Set camera intrinsics matrix

        Args:
            leftCameraMatrix (np.ndarray): Left camera intrinsics matrix in opencv format
            rightCameraMatrix (np.ndarray): Right camera intrinsics matrix in opencv format
        """
        self.leftCameraMatrix = leftCameraMatrix
        self.rightCameraMatrix = rightCameraMatrix


    def setDistortionCoefficients(self, leftDistCoeffs: np.ndarray, rightDistCoeffs: np.ndarray):
        """Set camera distortion coefficients

        Args:
            leftDistCoeffs (np.ndarray): Left camera distortion coefficient matrix in opencv format
            rightDistCoeffs (np.ndarray): Right camera distortion coefficient matrix in opencv format
        """
        self.leftDistCoeffs = leftDistCoeffs
        self.rightDistCoeffs = rightDistCoeffs


    def setTargetModel(self, targetModel: TargetModel):
        """Set target model to use

        Args:
            targetModel (TargetModel): Target model expected to be seen
        """
        self.targetModel = targetModel
        self.size = targetModel.getVertices()[0].distance(targetModel.getVertices()[1])


    def setAprilTagFieldLayout(self, fieldLayout: AprilTagFieldLayout):
        """Set AprilTag field layout

        Args:
            fieldLayout (AprilTagFieldLayout): Field layout of tags
        """
        self.fieldLayout = fieldLayout


    def setTagBlacklist(self, blacklist: list[int]):
        """Set blacklist of tags

        Args:
            blacklist (list[int]): Tags to NOT usel
        """
        self.tagBlacklist = blacklist
