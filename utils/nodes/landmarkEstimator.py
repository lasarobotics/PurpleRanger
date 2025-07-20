import logging
import marshal

import depthai

import numpy as np

from robotpy_apriltag import AprilTagFieldLayout
from wpimath.geometry import Transform3d, Pose3d

from utils.apriltag import OpenCVHelp, TargetModel, TagCorner, AprilTagPoseEstimation

class LandmarkEstimator(depthai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.tags = self.createInput()
        self.landmarks = self.createOutput()


    def run(self):
        while self.isRunning():
            input_buffer = self.tags.get() # Get a buffer from the input queue
            output_buffer = depthai.Landmarks()
            visible_landmarks = []
            self.tagBlacklist = []

            result = AprilTagPoseEstimation.estimateCamPosePNP(self.cameraMatrix, self.distCoeffs, input_buffer.aprilTags, self.fieldLayout, self.targetModel)
            if result:
                for tagID in result.fiducialIDsUsed:
                    result_pose = Pose3d(result.best.translation(), result.best.rotation())
                    camToTag = Transform3d(result_pose, self.fieldLayout.getTagPose(tagID))
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

                    visible_landmarks.append(landmark)

            output_buffer.setTimestamp(input_buffer.getTimestamp())
            output_buffer.setTimestampDevice(input_buffer.getTimestampDevice())
            output_buffer.landmarks = visible_landmarks
            self.landmarks.send(output_buffer)


    def setCameraIntrinsics(self, cameraMatrix: np.ndarray):
        """Set camera intrinsics matrix

        Args:
            cameraMatrix (np.ndarray): Camera intrinsics matrix in opencv format
        """
        self.cameraMatrix = cameraMatrix


    def setDistortionCoefficients(self, distCoeffs: np.ndarray):
        """Set camera distortion coefficients

        Args:
            distCoeffs (np.ndarray): Camera distortion coefficient matrix in opencv format
        """
        self.distCoeffs = distCoeffs


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