import depthai

import math

import numpy as np
from robotpy_apriltag import AprilTag, AprilTagFieldLayout
from wpimath.geometry import Pose3d, Transform3d, Translation3d, Rotation3d

from .multiTargetPNPResult import PnpResult
from .tagCorner import TagCorner
from . import OpenCVHelp, TargetModel

TAG_TRANSFORM = Transform3d(Translation3d(), Rotation3d(math.pi, 0.0, math.pi))

class AprilTagPoseEstimation:

    @staticmethod
    def estimateCamPosePNP(
        cameraMatrix: np.ndarray,
        distCoeffs: np.ndarray,
        visibleTags: list[depthai.AprilTag],
        layout: AprilTagFieldLayout,
        tagModel: TargetModel) -> PnpResult | None:
        """Performs solvePNP using 3d-2d point correspondences of visible AprilTags to estimate the
        field-to-camera transformation. If only one tag is visible, the result may have an alternate
        solution.

        **Note:** The returned transformation is from the field origin to the camera pose!

        With only one tag: {@link OpenCVHelp#solvePNP_SQUARE}

        With multiple tags: {@link OpenCVHelp#solvePNP_SQPNP}

        :param cameraMatrix: The camera intrinsics matrix in standard opencv form
        :param distCoeffs:   The camera distortion matrix in standard opencv form
        :param visTags:      The visible tags reported by PV. Non-tag targets are automatically excluded.
        :param tagLayout:    The known tag layout on the field

        :returns: The transformation that maps the field origin to the camera pose. Ensure the {@link
                  PnpResult} are present before utilizing them.
        """

        if len(visibleTags) == 0:
            return None

        corners: list[TagCorner] = []
        knownTags: list[AprilTag] = []

        # ensure these are AprilTags in our layout
        for target in visibleTags:
            maybePose = layout.getTagPose(target.id)
            if maybePose:
                tag = AprilTag()
                tag.ID = target.id
                tag.pose = maybePose.transformBy(TAG_TRANSFORM)
                knownTags.append(tag)
                currentCorners = [
                    TagCorner(target.topLeft.x, target.topLeft.y),
                    TagCorner(target.topRight.x, target.topRight.y),
                    TagCorner(target.bottomRight.x, target.bottomRight.y),
                    TagCorner(target.bottomLeft.x, target.bottomLeft.y)
                ]
                if currentCorners:
                    corners += currentCorners

        if len(knownTags) == 0 or len(corners) == 0 or len(corners) % 4 != 0:
            return None

        points = OpenCVHelp.cornersToPoints(corners)

        # single-tag pnp
        if len(knownTags) == 1:
            camToTag = OpenCVHelp.solvePNP_Square(cameraMatrix, distCoeffs, tagModel.getVertices(), points)
            if not camToTag:
                return None

            bestPose = knownTags[0].pose.transformBy(camToTag.best.inverse())
            altPose = Pose3d()
            if camToTag.ambiguity != 0:
                altPose = knownTags[0].pose.transformBy(camToTag.alt.inverse())

            o = Pose3d()
            result = PnpResult(
                best=Transform3d(o, bestPose),
                alt=Transform3d(o, altPose),
                ambiguity=camToTag.ambiguity,
                bestReprojErr=camToTag.bestReprojErr,
                altReprojErr=camToTag.altReprojErr,
            )
            return result
        # multi-tag pnp
        else:
            objectTrls: list[Translation3d] = []
            for tag in knownTags:
                verts = tagModel.getFieldVertices(tag.pose)
                objectTrls += verts

            ret = OpenCVHelp.solvePNP_SQPNP(cameraMatrix, distCoeffs, objectTrls, points)
            if ret:
                # Invert best/alt transforms
                ret.best = ret.best.inverse()
                ret.alt = ret.alt.inverse()

            return ret
