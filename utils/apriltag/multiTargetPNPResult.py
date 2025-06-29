from dataclasses import dataclass, field
from wpimath.geometry import Transform3d

@dataclass
class PnpResult:
    best: Transform3d = field(default_factory=Transform3d)
    alt: Transform3d = field(default_factory=Transform3d)
    ambiguity: float = 0.0
    bestReprojErr: float = 0.0
    altReprojErr: float = 0.0


@dataclass
class MultiTargetPNPResult:
    _MAX_IDS = 32

    estimatedPose: PnpResult = field(default_factory=PnpResult)
    fiducialIDsUsed: list[int] = field(default_factory=list)
