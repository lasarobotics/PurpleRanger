import logging
import marshal

import depthai

import numpy as np

from robotpy_apriltag import AprilTagFieldLayout

from utils.apriltag import OpenCVHelp, TargetModel, TagCorner

class TagLandmarkEstimator(depthai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.tags = self.createInput()
        self.landmarks = self.createOutput()
        self.size = 6.5 * 0.0254
        self.tagBlacklist = []

    def run(self):
        while self.isRunning():
            input_buffer = self.tags.get() # Get a buffer from the input queue
            output_buffer = depthai.Landmarks()
            visible_landmarks = []

            for target in input_buffer.aprilTags:
                if target.id in self.tagBlacklist: continue
                maybePose = self.fieldLayout.getTagPose(target.id)
                if maybePose:
                    corners = [
                        TagCorner(target.topLeft.x, target.topLeft.y),
                        TagCorner(target.topRight.x, target.topRight.y),
                        TagCorner(target.bottomRight.x, target.bottomRight.y),
                        TagCorner(target.bottomLeft.x, target.bottomLeft.y)
                    ]
                    points = OpenCVHelp.cornersToPoints(corners)
                    camToTag = OpenCVHelp.solvePNP_Square(self.cameraMatrix, self.distCoeffs, self.targetModel.getVertices(), points)\

                    landmark = depthai.Landmark()
                    landmark.id = target.id
                    landmark.size = self.size
                    landmark.translation.x = camToTag.best.translation().X()
                    landmark.translation.y = camToTag.best.translation().Y()
                    landmark.translation.z = camToTag.best.translation().Z()
                    landmark.quaternion.qx = camToTag.best.rotation().getQuaternion().X()
                    landmark.quaternion.qy = camToTag.best.rotation().getQuaternion().Y()
                    landmark.quaternion.qz = camToTag.best.rotation().getQuaternion().Z()
                    landmark.quaternion.qw = camToTag.best.rotation().getQuaternion().W()

                    visible_landmarks.append(landmark)

            output_buffer.setTimestamp(input_buffer.getTimestamp())
            output_buffer.setTimestampDevice(input_buffer.getTimestampDevice())
            output_buffer.landmarks = visible_landmarks
            logging.debug(str(output_buffer))
            self.landmarks.send(output_buffer)


    def setCameraExtrinsics(self, cameraMatrix: np.ndarray):
        self.cameraMatrix = cameraMatrix


    def setDistortionCoefficients(self, distCoeffs: np.ndarray):
        self.distCoeffs = distCoeffs


    def setTargetModel(self, targetModel: TargetModel):
        self.targetModel = targetModel
        self.size = targetModel.getVertices()[0].distance(targetModel.getVertices()[1])


    def setAprilTagFieldLayout(self, fieldLayout: AprilTagFieldLayout):
        self.fieldLayout = fieldLayout


    def setTagBlacklist(self, blacklist: list[int]):
        self.tagBlacklist = blacklist