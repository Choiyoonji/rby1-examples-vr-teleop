import pickle
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(slots=True)
class ControlState:

    timestamp: float = 0.0

    joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_currents: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_torques: np.ndarray = field(default_factory=lambda: np.array([]))
    right_force_sensor: np.ndarray = field(default_factory=lambda: np.array([]))
    left_force_sensor: np.ndarray = field(default_factory=lambda: np.array([]))
    right_torque_sensor: np.ndarray = field(default_factory=lambda: np.array([]))
    left_torque_sensor: np.ndarray = field(default_factory=lambda: np.array([]))
    center_of_mass: np.ndarray = field(default_factory=lambda: np.array([]))
    command: dict = field(default_factory=dict)

    # Flags
    is_initialized: bool = False
    is_stopped: bool = False

    # Mobile base velocities
    mobile_linear_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    mobile_angular_velocity: float = 0.0

    # Following state
    is_torso_following: bool = False
    is_right_following: bool = False
    is_left_following: bool = False

    # Base pose
    base_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    base_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))

    # Head command
    head_command_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    head_command_current_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    torso_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    torso_current_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    torso_locked_pose: np.ndarray = field(default_factory=lambda: np.identity(4))

    # Right arm command & EE
    right_command_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    right_command_current_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    right_ee_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    right_ee_current_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    right_arm_locked_pose: np.ndarray = field(default_factory=lambda: np.identity(4))

    # Left arm command & EE
    left_command_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    left_command_current_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    left_ee_start_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    left_ee_current_pose: np.ndarray = field(default_factory=lambda: np.identity(4))
    left_arm_locked_pose: np.ndarray = field(default_factory=lambda: np.identity(4))

    # Move and Stop events
    ready: bool = False
    move: bool = False
    stop: bool = False