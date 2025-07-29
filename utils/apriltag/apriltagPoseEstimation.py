import sys
import math
import logging
from enum import Enum

import numpy as np

import depthai

from robotpy_apriltag import AprilTag, AprilTagFieldLayout
from wpimath.geometry import Pose3d, Twist3d, Transform3d, Translation3d, Rotation3d

from .multiTargetPNPResult import PnpResult
from .tagCorner import TagCorner
from . import OpenCVHelp, TargetModel

TAG_TRANSFORM = Transform3d(Translation3d(), Rotation3d(math.pi, 0.0, math.pi))
ROLL_THRESHOLD = math.radians(60.0)
PITCH_THRESHOLD = math.radians(60.0)

DEFAULT_TAG_MEASUREMENT_NOISE = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]

class Perspective(Enum):
    LEFT = 0
    RIGHT = 1
    CENTER = 2

class AprilTagPoseEstimation:

    @staticmethod
    def mergePoses(left_estimate: PnpResult, right_estimate: PnpResult, field_layout: AprilTagFieldLayout, baseline: float, perspective: Perspective = Perspective.CENTER) -> tuple[Pose3d, np.array]:
        left_transform = Transform3d(0, +baseline / 2, 0, Rotation3d())
        right_transform = Transform3d(0, -baseline / 2, 0, Rotation3d())

        match perspective:
            case Perspective.LEFT:
                left_transform = Transform3d(0, 0, 0, Rotation3d())
                right_transform = Transform3d(0, -baseline, 0, Rotation3d())
            case Perspective.RIGHT:
                left_transform = Transform3d(0, +baseline, 0, Rotation3d())
                right_transform = Transform3d(0, 0, 0, Rotation3d())
            case _:
                pass

        pose = None
        tags_used = []
        if left_estimate is None and right_estimate is None:
            logging.debug("No tags seen")
        elif left_estimate is not None and right_estimate is not None:
            tags_used = list(set(left_estimate.fiducialIDsUsed) & set(right_estimate.fiducialIDsUsed))
            left_offset = left_transform.inverse().translation().rotateBy(right_estimate.best.rotation())
            right_offset = right_transform.inverse().translation().rotateBy(right_estimate.best.rotation())
            left_pose = Pose3d(left_estimate.best.translation() + left_offset, left_estimate.best.rotation())
            right_pose = Pose3d(right_estimate.best.translation() + right_offset, right_estimate.best.rotation())
            twist = left_pose.log(right_pose)
            scaled_twist = Twist3d(
                twist.dx / 2, twist.dy / 2, twist.dz / 2,
                twist.rx / 2, twist.ry / 2, twist.rz / 2
            )
            pose = left_pose.exp(scaled_twist)
        elif left_estimate is None:
            tags_used = right_estimate.fiducialIDsUsed
            right_offset = right_transform.inverse().translation().rotateBy(right_estimate.best.rotation())
            pose = Pose3d(right_estimate.best.translation() + right_offset, right_estimate.best.rotation())
        elif right_estimate is None:
            tags_used = left_estimate.fiducialIDsUsed
            left_offset = left_transform.inverse().translation().rotateBy(left_estimate.best.rotation())
            pose = Pose3d(left_estimate.best.translation() + left_offset, left_estimate.best.rotation())

        minimum_distance = sys.maxsize
        for tag_id in tags_used:
            distance_to_tag = pose.translation().distance(field_layout.getTagPose(tag_id).translation())
            if distance_to_tag < minimum_distance: minimum_distance = distance_to_tag

        measurement_noise_std = DEFAULT_TAG_MEASUREMENT_NOISE
        if len(tags_used) > 1:
            translation_noise_std = 0.01 * (minimum_distance ** 2) / len(tags_used)
            measurement_noise_std = [translation_noise_std, translation_noise_std, translation_noise_std, 0.2, 0.2, 0.2]

        return pose, np.array(measurement_noise_std)

    @staticmethod
    def isResultValid(result: PnpResult) -> PnpResult:
        # if abs(result.best.rotation().X()) > ROLL_THRESHOLD or abs(result.best.rotation().Y() > PITCH_THRESHOLD):
        #     return None

        return result

    @staticmethod
    def estimateCamPosePNP(
        cameraMatrix: np.ndarray,
        distCoeffs: np.ndarray,
        visibleTags: list[depthai.AprilTag],
        layout: AprilTagFieldLayout,
        tagModel: TargetModel,
        tagBlacklist: list[int] = []) -> PnpResult | None:
        """Performs solvePNP using 3d-2d point correspondences of visible AprilTags to estimate the
        field-to-camera transformation. If only one tag is visible, the result may have an alternate
        solution.

        **Note:** The returned transformation is from the field origin to the camera pose!

        With only one tag: {@link OpenCVHelp#solvePNP_SQUARE}

        With multiple tags: {@link OpenCVHelp#solvePNP_SQPNP}

        :param cameraMatrix: The camera intrinsics matrix in standard opencv form
        :param distCoeffs:   The camera distortion matrix in standard opencv form
        :param visibleTags:      The visible tags reported by PV. Non-tag targets are automatically excluded.
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
            if target.id in tagBlacklist: continue
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
                fiducialIDsUsed=[knownTags[0].ID]
            )
            return AprilTagPoseEstimation.isResultValid(result)

        # multi-tag pnp
        else:
            objectTrls: list[Translation3d] = []
            for tag in knownTags:
                verts = tagModel.getFieldVertices(tag.pose)
                objectTrls += verts

            result = OpenCVHelp.solvePNP_SQPNP(cameraMatrix, distCoeffs, objectTrls, points)
            if result:
                # Invert best/alt transforms
                result.best = result.best.inverse()
                result.alt = result.alt.inverse()
                result.fiducialIDsUsed = [tag.ID for tag in knownTags]

            return AprilTagPoseEstimation.isResultValid(result)

