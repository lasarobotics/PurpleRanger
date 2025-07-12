import depthai

import numpy as np

from robotpy_apriltag import AprilTagFieldLayout

from utils.apriltag import TargetModel

class TagLandmarkEstimator(depthai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        self.tags = self.createInput()
        self.landmarks = self.createOutput()

    def onStart(self):
        pass

    def run(self):
        while self.isRunning():
            input_buffer = self.tags.get() # Get a buffer from the input queue
            output_buffer = depthai.Buffer()
            landmarks = []
            for target in input_buffer:
                if target.id in tagBlacklist: continue
                maybePose = self.fieldLayout.getTagPose(target.id)
                if maybePose:
                    corners = [
                        TagCorner(target.topLeft.x, target.topLeft.y),
                        TagCorner(target.topRight.x, target.topRight.y),
                        TagCorner(target.bottomRight.x, target.bottomRight.y),
                        TagCorner(target.bottomLeft.x, target.bottomLeft.y)
                    ]
                    points = OpenCVHelp.cornersToPoints(corners)
                    camToTag = OpenCVHelp.solvePNP_Square(self.cameraMatrix, self.distCoeffs, self.targetModel.getVertices(), points)
                    transform = TransformData(camToTag.best.translation().X(), camToTag.best.translation().Y(), camToTag.best.translation().Z(), camToTag.best.rotation().X(), camToTag.best.rotation().Y(), camToTag.best.rotation.Z())
                    landmarks.append(depthai.Landmark(target.id, self.size, transform))

            output_buffer.setData(landmarks)
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