# -*- coding: utf-8 -*-
"""Pure, fail-closed PC-side gate for the ESTUN P2/P1 CRP barriers.

No ticket is put on the ESTUN wire.  This module binds local tickets to one
TCP run, its plan, full hand-eye calibration digest and a post-P2 wrist
observation.  It intentionally has no socket, camera or robot imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import math
import secrets
import struct
import time


class BarrierReject(Exception):
    """Terminal protocol error.  The caller must close the TCP session."""

    def __init__(self, reason: str, code: str = "REJECT") -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code


class SessionState(StrEnum):
    WAIT_P2_ARRIVED = "WAIT_P2_ARRIVED"
    WAIT_FRESH_WRIST_CONFIRMATION = "WAIT_FRESH_WRIST_CONFIRMATION"
    WAIT_P1_ARRIVED = "WAIT_P1_ARRIVED"
    WAIT_GRIPDONE = "WAIT_GRIPDONE"
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class WristObservation:
    """A stable, locally generated wrist observation; never controller input."""

    observation_id: int
    observed_at_ns: int

    def __post_init__(self) -> None:
        if isinstance(self.observation_id, bool) or not isinstance(self.observation_id, int):
            raise ValueError("observation_id must be an integer")
        _require_ns(self.observed_at_ns)


def _require_ns(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("monotonic timestamp must be a non-negative integer")


def _fault(reason: str, code: str) -> None:
    raise BarrierReject(reason, code)


def parse_arrived(msg: object, prefix: str):
    """Parse exactly ``PREFIX|x,y,z,a,b,c`` in metres/degrees."""
    if not isinstance(msg, str):
        _fault("message is not a string", "NOT_STR")
    if not msg:
        _fault("empty message", "EMPTY")
    if not msg.isascii():
        _fault("message contains non-ASCII characters", "NON_ASCII")
    if any(char.isspace() for char in msg):
        _fault("message contains whitespace", "WHITESPACE")
    if not msg.startswith(prefix):
        _fault(f"expected prefix {prefix!r}", "BAD_PREFIX")
    body = msg[len(prefix):]
    parts = body.split(",")
    if len(parts) != 6 or any(not part for part in parts):
        _fault("CPOS must have six non-empty fields", "BAD_COUNT")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        _fault("CPOS contains a non-numeric field", "BAD_NUMBER")
    if not all(math.isfinite(value) for value in values):
        _fault("CPOS contains a non-finite field", "NOT_FINITE")
    return values[:3], values[3:]


def _angular_distance_deg(left: float, right: float) -> float:
    raw = abs(left - right) % 360.0
    return min(raw, 360.0 - raw)


def _pose_matches(actual, expected, *, position_tolerance_m: float, angle_tolerance_deg: float) -> bool:
    position_error = math.sqrt(sum((actual - target) ** 2 for actual, target in zip(actual[0], expected[0])))
    return position_error <= position_tolerance_m and all(
        _angular_distance_deg(actual_angle, target_angle) <= angle_tolerance_deg
        for actual_angle, target_angle in zip(actual[1], expected[1])
    )


def _normalise_pose(pose, *, label: str):
    try:
        xyz, abc = pose
        values = tuple(float(value) for value in (*xyz, *abc))
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain xyz and abc triplets") from None
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain six finite values")
    return values[:3], values[3:]


def calib_fingerprint(matrix) -> str:
    """Digest every finite entry in a 4×4 transform, including rotation."""
    try:
        rows = tuple(tuple(float(value) for value in row) for row in matrix)
    except (TypeError, ValueError):
        raise ValueError("calibration matrix must be a numeric 4x4 matrix") from None
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("calibration matrix must be 4x4")
    values = tuple(value for row in rows for value in row)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("calibration matrix contains a non-finite value")
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("!d", value))
    return digest.hexdigest()


class SessionBarrier:
    """One TCP run.  Instantiate once, call only from that run's handler."""

    def __init__(
        self,
        run_id: str,
        *,
        lease_epoch: int,
        ticket_ttl_s: float = 30.0,
        gripdone_timeout_s: float = 10.0,
        position_tolerance_m: float = 0.02,
        angle_tolerance_deg: float = 5.0,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be non-empty text")
        if isinstance(lease_epoch, bool) or not isinstance(lease_epoch, int) or lease_epoch < 1:
            raise ValueError("lease_epoch must be a positive integer")
        for name, value in (
            ("ticket_ttl_s", ticket_ttl_s),
            ("gripdone_timeout_s", gripdone_timeout_s),
            ("position_tolerance_m", position_tolerance_m),
            ("angle_tolerance_deg", angle_tolerance_deg),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self.run_id = run_id
        self.lease_epoch = lease_epoch
        self._ticket_ttl_ns = int(ticket_ttl_s * 1_000_000_000)
        self._gripdone_timeout_ns = int(gripdone_timeout_s * 1_000_000_000)
        self._position_tolerance_m = float(position_tolerance_m)
        self._angle_tolerance_deg = float(angle_tolerance_deg)
        self.state = SessionState.WAIT_P2_ARRIVED
        self._p0_pose = None
        self._p2_target = None
        self._p1_target = None
        self._planning_observation_id = None
        self._calibration_fingerprint = None
        self._plan_digest = None
        self._pregrasp_ticket = None
        self._grasp_ticket = None
        self._p2_arrived_at_ns = None
        self._gripdone_deadline_ns = None

    @property
    def p2_arrived_at_ns(self) -> int:
        if self._p2_arrived_at_ns is None:
            raise RuntimeError("P2 has not been received")
        return self._p2_arrived_at_ns

    def arm(
        self,
        *,
        p0_pose,
        p2_target,
        p1_target,
        planning_observation_id: int,
        calibration_fingerprint: str,
        now_ns: int | None = None,
    ) -> None:
        now_ns = self._now(now_ns)
        if self._pregrasp_ticket is not None or self.state is not SessionState.WAIT_P2_ARRIVED:
            self._reject("session already armed", "STATE")
        self._p0_pose = _normalise_pose(p0_pose, label="P0 pose")
        if all(value == 0.0 for value in (*self._p0_pose[0], *self._p0_pose[1])):
            self._reject("P0 pose must not be all zero", "BAD_P0")
        self._p2_target = _normalise_pose(p2_target, label="P2 target")
        self._p1_target = _normalise_pose(p1_target, label="P1 target")
        if isinstance(planning_observation_id, bool) or not isinstance(planning_observation_id, int):
            self._reject("planning observation ID is invalid", "OBS_ID")
        if not isinstance(calibration_fingerprint, str) or len(calibration_fingerprint) != 64:
            self._reject("calibration fingerprint is invalid", "CALIB")
        self._planning_observation_id = planning_observation_id
        self._calibration_fingerprint = calibration_fingerprint
        self._plan_digest = self._digest_plan()
        self._pregrasp_ticket = self._new_ticket("pregrasp", now_ns)

    def receive_p2(self, raw: object, *, calibration_fingerprint: str, now_ns: int | None = None) -> None:
        now_ns = self._now(now_ns)
        self._require_state(SessionState.WAIT_P2_ARRIVED)
        actual = self._parse_or_reject(raw, "P2_ARRIVED|")
        self._require_calibration(calibration_fingerprint)
        self._require_ticket(self._pregrasp_ticket, now_ns)
        if not _pose_matches(
            actual,
            self._p2_target,
            position_tolerance_m=self._position_tolerance_m,
            angle_tolerance_deg=self._angle_tolerance_deg,
        ):
            self._reject("P2 feedback exceeds tolerance", "P2_TOL")
        self._p2_arrived_at_ns = now_ns
        self.state = SessionState.WAIT_FRESH_WRIST_CONFIRMATION

    def authorize_grasp(
        self,
        observation: WristObservation,
        *,
        calibration_fingerprint: str,
        now_ns: int | None = None,
    ) -> str:
        now_ns = self._now(now_ns)
        self._require_state(SessionState.WAIT_FRESH_WRIST_CONFIRMATION)
        self._require_calibration(calibration_fingerprint)
        if not isinstance(observation, WristObservation):
            self._reject("fresh wrist observation is invalid", "OBS")
        if observation.observed_at_ns > now_ns:
            self._reject("fresh wrist observation is in the future", "OBS_FUTURE")
        if self._p2_arrived_at_ns is None or observation.observed_at_ns <= self._p2_arrived_at_ns:
            self._reject("fresh wrist observation must be after P2", "STALE_OBS")
        if observation.observation_id == self._planning_observation_id:
            self._reject("fresh wrist observation reused the planning observation", "STALE_OBS")
        self._grasp_ticket = self._new_ticket("grasp", now_ns)
        self.state = SessionState.WAIT_P1_ARRIVED
        return self._ack("<", ">")

    def receive_p1(self, raw: object, *, calibration_fingerprint: str, now_ns: int | None = None) -> str:
        now_ns = self._now(now_ns)
        self._require_state(SessionState.WAIT_P1_ARRIVED)
        actual = self._parse_or_reject(raw, "P1_ARRIVED|")
        self._require_calibration(calibration_fingerprint)
        self._require_ticket(self._grasp_ticket, now_ns)
        if not _pose_matches(
            actual,
            self._p1_target,
            position_tolerance_m=self._position_tolerance_m,
            angle_tolerance_deg=self._angle_tolerance_deg,
        ):
            self._reject("P1 feedback exceeds tolerance", "P1_TOL")
        self._grasp_ticket["consumed"] = True
        self._gripdone_deadline_ns = now_ns + self._gripdone_timeout_ns
        self.state = SessionState.WAIT_GRIPDONE
        return self._ack("[", "]")

    def receive_gripdone(self, raw: object, *, now_ns: int | None = None) -> None:
        now_ns = self._now(now_ns)
        self._require_state(SessionState.WAIT_GRIPDONE)
        self._require_gripdone_deadline(now_ns)
        if raw != "GRIPDONE":
            self._reject("expected exact GRIPDONE", "SEQ")
        self.state = SessionState.COMPLETE

    def check_timeout(self, *, now_ns: int | None = None) -> None:
        now_ns = self._now(now_ns)
        if self.state is SessionState.WAIT_GRIPDONE:
            self._require_gripdone_deadline(now_ns)

    def _now(self, now_ns: int | None) -> int:
        value = time.monotonic_ns() if now_ns is None else now_ns
        _require_ns(value)
        return value

    def _parse_or_reject(self, raw: object, prefix: str):
        try:
            return parse_arrived(raw, prefix)
        except BarrierReject as exc:
            self._reject(exc.reason, exc.code)

    def _require_state(self, expected: SessionState) -> None:
        if self.state is SessionState.REJECTED:
            raise BarrierReject("session is already rejected", "STATE")
        if self.state is not expected:
            self._reject(f"unexpected state {self.state}", "SEQ")

    def _require_calibration(self, current: str) -> None:
        if not isinstance(current, str) or current != self._calibration_fingerprint:
            self._reject("calibration fingerprint changed", "CALIB")

    def _require_ticket(self, ticket, now_ns: int) -> None:
        if ticket is None or ticket.get("consumed"):
            self._reject("ticket is absent or consumed", "REPLAY")
        if ticket.get("scope") != self._plan_digest:
            self._reject("ticket scope mismatch", "SCOPE")
        if now_ns > ticket["expires_at_ns"]:
            self._reject("ticket timeout", "TICKET_TIMEOUT")

    def _require_gripdone_deadline(self, now_ns: int) -> None:
        if self._gripdone_deadline_ns is None or now_ns > self._gripdone_deadline_ns:
            self._reject("GRIPDONE timeout", "GRIPDONE_TIMEOUT")

    def _new_ticket(self, kind: str, now_ns: int):
        return {
            "id": f"{kind}-{self.run_id}-{secrets.token_hex(12)}",
            "scope": self._plan_digest,
            "expires_at_ns": now_ns + self._ticket_ttl_ns,
            "consumed": False,
        }

    def _digest_plan(self) -> str:
        assert self._p2_target is not None and self._p1_target is not None
        assert self._calibration_fingerprint is not None
        text = repr((self.run_id, self.lease_epoch, self._calibration_fingerprint, self._p2_target, self._p1_target))
        return hashlib.sha256(text.encode("ascii")).hexdigest()

    def _ack(self, left: str, right: str) -> str:
        assert self._p0_pose is not None
        xyz, abc = self._p0_pose
        return f"{left}{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f},{abc[0]:.2f},{abc[1]:.2f},{abc[2]:.2f}{right}"

    def _reject(self, reason: str, code: str) -> None:
        self.state = SessionState.REJECTED
        if self._pregrasp_ticket is not None:
            self._pregrasp_ticket["consumed"] = True
        if self._grasp_ticket is not None:
            self._grasp_ticket["consumed"] = True
        raise BarrierReject(reason, code)
