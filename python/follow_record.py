"""
python follow_record.py \
  --local_ip 127.0.0.1 --local_port 6001 \
  --control_ip 127.0.0.1 --control_port 6002 \
  --rby1 192.168.0.83:50051 --rby1_model a \
  --no_gripper
"""

import argparse
import logging
import zmq
import time
import threading
from dataclasses import dataclass
import rby1_sdk as rby
import socket
from typing import Union
import json
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from gripper import Gripper
import pickle
from control_state import ControlState

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)-8s - %(message)s"
)

T_conv = np.array([
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
])


class Settings:
    dt: float = 0.1  # 10 Hz
    update_dt : float = 0.01   # 100 Hz
    hand_offset: float = np.array([0.0, 0.0, 0.0])

    T_hand_offset = np.identity(4)
    T_hand_offset[0:3, 3] = hand_offset

    local_ip = "127.0.0.1"
    local_port = 5555
    control_ip = "127.0.0.1"
    control_port = 5556

    mobile_linear_acceleration_gain: float = 0.15
    mobile_angular_acceleration_gain: float = 0.15
    mobile_linear_damping_gain: float = 0.3
    mobile_angular_damping_gain: float = 0.3


class SystemContext:
    robot_model: Union[rby.Model_A, rby.Model_M] = None
    control_state = ControlState()


def robot_state_callback(robot_state: rby.RobotState_A):
    SystemContext.control_state.timestamp = robot_state.timestamp.timestamp()
    SystemContext.control_state.joint_positions = robot_state.position
    SystemContext.control_state.joint_velocities = robot_state.velocity
    SystemContext.control_state.joint_currents = robot_state.current
    SystemContext.control_state.joint_torques = robot_state.torque
    SystemContext.control_state.right_force_sensor = robot_state.ft_sensor_right.force
    SystemContext.control_state.left_force_sensor = robot_state.ft_sensor_left.force
    SystemContext.control_state.right_torque_sensor = robot_state.ft_sensor_right.torque
    SystemContext.control_state.left_torque_sensor = robot_state.ft_sensor_left.torque
    SystemContext.control_state.center_of_mass = robot_state.center_of_mass


def connect_rby1(address: str, model: str = "a", no_head: bool = False):
    logging.info(f"Attempting to connect to RB-Y1... (Address: {address}, Model: {model})")
    robot = rby.create_robot(address, model)

    connected = robot.connect()
    if not connected:
        logging.critical("Failed to connect to RB-Y1. Exiting program.")
        exit(1)

    logging.info("Successfully connected to RB-Y1.")

    servo_pattern = "^(?!head_).*" if no_head else ".*"
    if not robot.is_power_on(servo_pattern):
        logging.warning("Robot power is off. Turning it on...")
        if not robot.power_on(servo_pattern):
            logging.critical("Failed to power on. Exiting program.")
            exit(1)
        logging.info("Power turned on successfully.")
    else:
        logging.info("Power is already on.")

    if not robot.is_servo_on(".*"):
        logging.warning("Servo is off. Turning it on...")
        if not robot.servo_on(".*"):
            logging.critical("Failed to turn on the servo. Exiting program.")
            exit(1)
        logging.info("Servo turned on successfully.")
    else:
        logging.info("Servo is already on.")

    cm_state = robot.get_control_manager_state().state
    if cm_state in [
        rby.ControlManagerState.State.MajorFault,
        rby.ControlManagerState.State.MinorFault,
    ]:
        logging.warning(f"Control Manager is in Fault state: {cm_state.name}. Attempting reset...")
        if not robot.reset_fault_control_manager():
            logging.critical("Failed to reset Control Manager. Exiting program.")
            exit(1)
        logging.info("Control Manager reset successfully.")
    if not robot.enable_control_manager(unlimited_mode_enabled=True):
        logging.critical("Failed to enable Control Manager. Exiting program.")
        exit(1)
    logging.info("Control Manager successfully enabled. (Unlimited Mode: enabled)")

    SystemContext.robot_model = robot.model()
    robot.start_state_update(robot_state_callback, 1 / Settings.update_dt)

    return robot


def setup_command_udp_receiver(local_ip: str, local_port: int, power_off=None):
    """외부 파이썬 컨트롤러가 보내는 JSON 명령을 수신한다."""
    def udp_server():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((local_ip, local_port))
            logging.info(f"[RX] UDP listening for controller commands at {local_ip}:{local_port}")

            while True:
                data, addr = server_sock.recvfrom(4096)
                udp_msg = data.decode('utf-8')
                try:
                    SystemContext.control_state.command = json.loads(udp_msg)
                    SystemContext.control_state.ready = SystemContext.control_state.command.get("ready", False)
                    SystemContext.control_state.move = SystemContext.control_state.command.get("move", False)
                    SystemContext.control_state.stop = SystemContext.control_state.command.get("stop", False)

                    if SystemContext.control_state.command.get("estop", False):
                        if power_off is not None:
                            logging.warning("Estop button pressed. Shutting down power.")
                            power_off()

                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to decode JSON: {e} (from {addr}) - received data: {message[:100]}")

    thread = threading.Thread(target=udp_server, daemon=True)
    thread.start()


def handle_vr_button_event(robot: Union[rby.Robot_A, rby.Robot_M], no_head: bool):
    # ready: Initialize / Move to ready pose
    if SystemContext.control_state.ready:
        logging.info("Ready button pressed. Moving robot to ready pose.")
        if robot.get_control_manager_state().control_state != rby.ControlManagerState.ControlState.Idle:
            robot.cancel_control()
        if robot.wait_for_control_ready(1000):
            ready_pose = np.deg2rad(
                [0.0, 45.0, -90.0, 45.0, 0.0, 0.0] +
                [0.0, -15.0, 0.0, -120.0, 0.0, 70.0, 0.0] +
                [0.0, 15.0, 0.0, -120.0, 0.0, 70.0, 0.0])
            cbc = (
                rby.ComponentBasedCommandBuilder()
                .set_body_command(
                    rby.JointImpedanceControlCommandBuilder()
                    .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1))
                    .set_position(ready_pose)
                    .set_stiffness([400.] * 6 + [60] * 7 + [60] * 7)
                    .set_torque_limit([500] * 6 + [30] * 7 + [30] * 7)
                    .set_minimum_time(2)
                )
            )
            if not no_head:
                cbc.set_head_command(
                    rby.JointPositionCommandBuilder()
                    .set_position([0.] * len(SystemContext.robot_model.head_idx))
                    .set_minimum_time(2)
                )
            robot.send_command(
                rby.RobotCommandBuilder().set_command(
                    cbc
                )
            ).get()
        SystemContext.control_state.is_initialized = True
        SystemContext.control_state.is_stopped = False

    # stop: Stop
    elif SystemContext.control_state.stop:
        logging.info("Stop button pressed. Stopping.")
        SystemContext.control_state.is_stopped = True

    else:
        return False # Stream not reset

    # Clear events
    SystemContext.control_state.ready = False
    SystemContext.control_state.stop = False

    return True # Stream reset


def pose_to_se3(position, rotation_quat):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(rotation_quat).as_matrix()
    T[:3, 3] = position
    return T


def average_so3_slerp(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    # 두 회전을 Rotation 객체로 변환
    rot1 = R.from_matrix(R1)
    rot2 = R.from_matrix(R2)

    # 보간 설정: t=0 => rot1, t=1 => rot2
    slerp = Slerp([0, 1], R.concatenate([rot1, rot2]))

    # 평균값은 중간지점 t=0.5
    rot_avg = slerp(0.5)
    return rot_avg.as_matrix()


# ---------------------- 통신: 송신(상태) ----------------------
def _mat4_to_list(T: np.ndarray):
    return T.tolist()

def _nd_to_list(x):
    return x.tolist() if isinstance(x, np.ndarray) else x

def pack_state_for_udp():
    cs = SystemContext.control_state
    return {
        "timestamp": cs.timestamp,

        "joint_positions": _nd_to_list(cs.joint_positions),
        "joint_velocities": _nd_to_list(cs.joint_velocities),
        "joint_currents": _nd_to_list(cs.joint_currents),
        "joint_torques": _nd_to_list(cs.joint_torques),

        "center_of_mass": _nd_to_list(cs.center_of_mass),

        "base_pose": _mat4_to_list(cs.base_pose),
        "torso_current_pose": _mat4_to_list(cs.torso_current_pose),
        "right_ee_current_pose": _mat4_to_list(cs.right_ee_current_pose),
        "left_ee_current_pose": _mat4_to_list(cs.left_ee_current_pose),

        "is_initialized": cs.is_initialized,
        "is_stopped": cs.is_stopped,
        "is_right_following": cs.is_right_following,
        "is_left_following": cs.is_left_following,
        "is_torso_following": cs.is_torso_following,

        "mobile_linear_velocity": _nd_to_list(cs.mobile_linear_velocity),
        "mobile_angular_velocity": cs.mobile_angular_velocity,
    }

def publish_state_udp(control_ip: str, control_port: int, period_s: float = 0.002):
    """로봇/컨트롤 상태를 주기적으로 컨트롤 PC에 UDP로 전송."""
    def loop():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx:
            target = (control_ip, control_port)
            logging.info(f"[TX] UDP state publisher -> {control_ip}:{control_port} (period={period_s}s)")
            while True:
                try:
                    payload = pack_state_for_udp()
                    blob = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                    tx.sendto(blob, target)
                    time.sleep(period_s)
                except Exception as e:
                    logging.exception(f"[TX] UDP send error: {e}")
                    time.sleep(period_s)
    t = threading.Thread(target=loop, daemon=True)
    t.start()


def main(args: argparse.Namespace):
    # Settings 반영
    Settings.local_ip = args.local_ip
    Settings.local_port = args.local_port
    Settings.control_ip = args.control_ip
    Settings.control_port = args.control_port

    logging.info("=== Controller-follow (UDP) Starting ===")
    logging.info(f"RX(bind)  : {Settings.local_ip}:{Settings.local_port}")
    logging.info(f"TX(target): {Settings.control_ip}:{Settings.control_port}")
    logging.info(f"RB-Y1     : {args.rby1} (model={args.rby1_model})")

    robot = connect_rby1(args.rby1, args.rby1_model, args.no_head)
    model = robot.model()

    # Start UDP Communication with Record
    # 명령 수신(컨트롤러 -> 본 프로세스)
    setup_command_udp_receiver(Settings.local_ip, Settings.local_port,
                               power_off=lambda: robot.power_off(".*"))

    # 상태 송신(본 프로세스 -> 컨트롤러)
    publish_state_udp(Settings.control_ip, Settings.control_port, period_s=Settings.update_dt)

    gripper = None
    if not args.no_gripper:
        for arm in ["left", "right"]:
            if not robot.set_tool_flange_output_voltage(arm, 12):
                logging.error(f"Failed to supply 12V to tool flange. ({arm})")
        time.sleep(0.5)
        gripper = Gripper()
        if not gripper.initialize(verbose=True):
            exit(1)
        gripper.homing()
        gripper.start()
        gripper.set_normalized_target(np.array([0.0, 0.0]))

    dyn_robot = robot.get_dynamics()
    dyn_state = dyn_robot.make_state(["base", "link_torso_5", "link_right_arm_6", "link_left_arm_6"],
                                     SystemContext.robot_model.robot_joint_names)
    base_link_idx, link_torso_5_idx, link_right_arm_6_idx, link_left_arm_6_idx = 0, 1, 2, 3

    next_time = time.monotonic()
    stream = None
    torso_reset = False
    right_reset = False
    left_reset = False

    while True:
        now = time.monotonic()
        if now < next_time:
            time.sleep(next_time - now)
        next_time += Settings.dt  # 10Hz

        if "arms" in SystemContext.control_state.command:
            if "right" in SystemContext.control_state.command["arms"]:
                right_controller = SystemContext.control_state.command["arms"]["right"]
                if gripper is not None:
                    gripper_target = gripper.get_normalized_target()
                    gripper_target[0] = right_controller["gripper"]
                    gripper.set_normalized_target(gripper_target)
            if "left" in SystemContext.control_state.command["arms"]:
                left_controller = SystemContext.control_state.command["arms"]["left"]
                if gripper is not None:
                    gripper_target = gripper.get_normalized_target()
                    gripper_target[1] = 1. - left_controller["gripper"]
                    gripper.set_normalized_target(gripper_target)

        if SystemContext.control_state.joint_positions.size == 0:
            continue

        # Ready / Stop button event handling -> reset stream
        if handle_vr_button_event(robot, args.no_head):
            if stream is not None:
                stream.cancel()
                stream = None

        if not SystemContext.control_state.is_initialized:
            continue

        if SystemContext.control_state.is_stopped:
            if stream is not None:
                stream.cancel()
                stream = None
            SystemContext.control_state.is_initialized = False
            continue

        logging.info(f"{SystemContext.control_state.center_of_mass = }")

        # Forward kinematics
        dyn_state.set_q(SystemContext.control_state.joint_positions.copy()) # Current Robot Joint Positions
        dyn_robot.compute_forward_kinematics(dyn_state)

        # Update current poses
        SystemContext.control_state.base_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx, link_torso_5_idx)
        SystemContext.control_state.torso_current_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx,
                                                                                     link_torso_5_idx)
        SystemContext.control_state.right_ee_current_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx,
                                                                                        link_right_arm_6_idx) @ Settings.T_hand_offset
        SystemContext.control_state.left_ee_current_pose = dyn_robot.compute_transformation(dyn_state, base_link_idx,
                                                                                       link_left_arm_6_idx) @ Settings.T_hand_offset

        # 12 = torso->right_ee, 13 = torso->left_ee
        trans_12 = dyn_robot.compute_transformation(dyn_state, 1, 2)
        trans_13 = dyn_robot.compute_transformation(dyn_state, 1, 3)
        center = (trans_12[:3, 3] + trans_13[:3, 3]) / 2
        yaw = np.atan2(center[1], center[0])
        pitch = np.atan2(-center[2], center[0]) - np.deg2rad(10)
        yaw = np.clip(yaw, -np.deg2rad(29), np.deg2rad(29))
        pitch = np.clip(pitch, -np.deg2rad(19), np.deg2rad(89))

        # Tracking Start - Initialize stream
        if stream is None:
            if robot.wait_for_control_ready(0):
                stream = robot.create_command_stream()
                SystemContext.control_state.mobile_linear_velocity = np.array([0.0, 0.0])
                SystemContext.control_state.mobile_angular_velocity = 0.
                SystemContext.control_state.is_right_following = False
                SystemContext.control_state.is_left_following = False
                SystemContext.control_state.base_start_pose = SystemContext.control_state.base_pose
                SystemContext.control_state.torso_locked_pose = SystemContext.control_state.torso_current_pose
                SystemContext.control_state.right_arm_locked_pose = SystemContext.control_state.right_ee_current_pose
                SystemContext.control_state.left_arm_locked_pose = SystemContext.control_state.left_ee_current_pose

        if "arms" in SystemContext.control_state.command:
            if "right" in SystemContext.control_state.command["arms"]:
                right_controller = SystemContext.control_state.command["arms"]["right"]

                # Update current pose
                SystemContext.control_state.right_command_current_pose = T_conv.T @ pose_to_se3(
                    right_controller["position"],
                    right_controller["rotation"]) @ T_conv

                move = SystemContext.control_state.move
                # If already following and move released -> stop following
                if SystemContext.control_state.is_right_following and not move:
                    SystemContext.control_state.is_right_following = False
                # If not following and move pressed -> start following
                if not SystemContext.control_state.is_right_following and move:
                    # Save current poses
                    SystemContext.control_state.right_command_start_pose = SystemContext.control_state.right_command_current_pose
                    SystemContext.control_state.right_ee_start_pose = SystemContext.control_state.right_ee_current_pose
                    SystemContext.control_state.is_right_following = True
                    right_reset = True
            # If no right controller data, stop following
            else:
                SystemContext.control_state.is_right_following = False

            # Same for left controller
            if "left" in SystemContext.control_state.command["arms"]:
                left_controller = SystemContext.control_state.command["arms"]["left"]

                SystemContext.control_state.left_command_current_pose = T_conv.T @ pose_to_se3(
                    left_controller["position"],
                    left_controller["rotation"]) @ T_conv

                move = SystemContext.control_state.move
                if SystemContext.control_state.is_left_following and not move:
                    SystemContext.control_state.is_left_following = False
                if not SystemContext.control_state.is_left_following and move:
                    SystemContext.control_state.left_command_start_pose = SystemContext.control_state.left_command_current_pose
                    SystemContext.control_state.left_ee_start_pose = SystemContext.control_state.left_ee_current_pose
                    SystemContext.control_state.is_left_following = True
                    left_reset = True
            else:
                SystemContext.control_state.is_left_following = False

            if "head" in SystemContext.control_state.command:
                head_controller = SystemContext.control_state.command["head"]
                SystemContext.control_state.head_command_current_pose = T_conv.T @ pose_to_se3(
                    head_controller["position"],
                    head_controller["rotation"]) @ T_conv

                # If both hands are following, start tracking head
                following = SystemContext.control_state.is_right_following and SystemContext.control_state.is_left_following
                if SystemContext.control_state.is_torso_following and not following:
                    SystemContext.control_state.is_torso_following = False
                if not SystemContext.control_state.is_torso_following and following:
                    SystemContext.control_state.head_command_start_pose = SystemContext.control_state.head_command_current_pose
                    SystemContext.control_state.torso_start_pose = SystemContext.control_state.torso_current_pose
                    SystemContext.control_state.is_torso_following = True
                    torso_reset = True
            else:
                SystemContext.control_state.is_torso_following = False

        SystemContext.control_state.mobile_linear_velocity -= Settings.mobile_linear_damping_gain * SystemContext.control_state.mobile_linear_velocity
        SystemContext.control_state.mobile_angular_velocity -= Settings.mobile_angular_damping_gain * SystemContext.control_state.mobile_angular_velocity

        if stream:
            try:
                if SystemContext.control_state.is_right_following:
                    # Compute the difference between the current and starting controller poses
                    # current = diff @ start -> diff = inv(start) @ current
                    diff = np.linalg.inv(
                        SystemContext.control_state.right_command_start_pose) @ SystemContext.control_state.right_command_current_pose

                    # Convert the difference to the global frame
                    T_global2start = np.identity(4)
                    T_global2start[:3, :3] = R.from_euler('y', 90, degrees=True).as_matrix() # Align axis
                    diff_global = T_global2start @ diff @ T_global2start.T

                    T = np.identity(4)
                    T[:3, :3] = SystemContext.control_state.right_ee_start_pose[:3, :3]
                    right_T = SystemContext.control_state.right_ee_start_pose @ diff_global
                    SystemContext.control_state.right_arm_locked_pose = right_T
                else:
                    right_T = SystemContext.control_state.right_arm_locked_pose

                if SystemContext.control_state.is_left_following:
                    diff = np.linalg.inv(
                        SystemContext.control_state.left_command_start_pose) @ SystemContext.control_state.left_command_current_pose

                    T_global2start = np.identity(4)
                    T_global2start[:3, :3] = R.from_euler('y', 90, degrees=True).as_matrix()
                    diff_global = T_global2start @ diff @ T_global2start.T

                    T = np.identity(4)
                    T[:3, :3] = SystemContext.control_state.left_ee_start_pose[:3, :3]
                    left_T = SystemContext.control_state.left_ee_start_pose @ diff_global
                    SystemContext.control_state.left_arm_locked_pose = left_T
                else:
                    left_T = SystemContext.control_state.left_arm_locked_pose

                if SystemContext.control_state.is_torso_following:
                    print('a')
                    diff = np.linalg.inv(
                        SystemContext.control_state.head_command_start_pose) @ SystemContext.control_state.head_command_current_pose
                    print(SystemContext.control_state.head_command_start_pose)

                    T = np.identity(4)
                    T[:3, :3] = SystemContext.control_state.torso_start_pose[:3, :3]
                    torso_T = SystemContext.control_state.torso_start_pose @ diff
                    SystemContext.control_state.torso_locked_pose = torso_T
                else:
                    torso_T = SystemContext.control_state.torso_locked_pose

                if args.whole_body:
                    ctrl_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([400.] * 6 + [60] * 7 + [60] * 7)
                        .set_joint_torque_limit([500] * 6 + [30] * 7 + [30] * 7)
                        .add_joint_limit("right_arm_3", -2.6, -0.5)
                        .add_joint_limit("right_arm_5", 0.2, 1.9)
                        .add_joint_limit("left_arm_3", -2.6, -0.5)
                        .add_joint_limit("left_arm_5", 0.2, 1.9)
                        .add_joint_limit("torso_1", -0.523598776, 1.3)
                        .add_joint_limit("torso_2", -2.617993878, -0.2)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(right_reset | left_reset | torso_reset)
                    )
                    ctrl_builder.add_target("base", "link_torso_5", torso_T, 1, np.pi * 0.5, 10, np.pi * 20)
                    ctrl_builder.add_target("base", "link_right_arm_6", right_T @ np.linalg.inv(Settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)
                    ctrl_builder.add_target("base", "link_left_arm_6", left_T @ np.linalg.inv(Settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)

                else:
                    torso_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([400.] * 6)
                        .set_joint_torque_limit([500] * 6)
                        .add_joint_limit("torso_1", -0.523598776, 1.3)
                        .add_joint_limit("torso_2", -2.617993878, -0.2)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(torso_reset)
                    )
                    right_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([80, 80, 80, 80, 80, 80, 40])
                        .set_joint_torque_limit([30] * 7)
                        .add_joint_limit("right_arm_3", -2.6, -0.5)
                        .add_joint_limit("right_arm_5", 0.2, 1.9)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(right_reset)
                    )
                    left_builder = (
                        rby.CartesianImpedanceControlCommandBuilder()
                        .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                        .set_minimum_time(Settings.dt * 1.01)
                        .set_joint_stiffness([80, 80, 80, 80, 80, 80, 40])
                        .set_joint_torque_limit([30] * 7)
                        .add_joint_limit("left_arm_3", -2.6, -0.5)
                        .add_joint_limit("left_arm_5", 0.2, 1.9)
                        .set_stop_joint_position_tracking_error(0)
                        .set_stop_orientation_tracking_error(0)
                        .set_stop_joint_position_tracking_error(0)
                        .set_reset_reference(left_reset)
                    )
                    torso_builder.add_target("base", "link_torso_5", torso_T, 1, np.pi * 0.5, 10, np.pi * 20)
                    right_builder.add_target("base", "link_right_arm_6",
                                             right_T @ np.linalg.inv(Settings.T_hand_offset),
                                             2, np.pi * 2, 20, np.pi * 80)
                    left_builder.add_target("base", "link_left_arm_6", left_T @ np.linalg.inv(Settings.T_hand_offset),
                                            2, np.pi * 2, 20, np.pi * 80)

                    ctrl_builder = (
                        rby.BodyComponentBasedCommandBuilder()
                        .set_torso_command(torso_builder)
                        .set_right_arm_command(right_builder)
                        .set_left_arm_command(left_builder)
                    )

                torso_reset = False
                right_reset = False
                left_reset = False

                stream.send_command(
                    rby.RobotCommandBuilder().set_command(
                        rby.ComponentBasedCommandBuilder()
                        .set_head_command(
                            rby.JointPositionCommandBuilder()
                            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                            .set_position([float(yaw), float(pitch)])
                            .set_minimum_time(Settings.dt * 1.01)
                        )
                        .set_mobility_command(
                            rby.SE2VelocityCommandBuilder()
                            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(Settings.dt * 10))
                            .set_velocity(-SystemContext.control_state.mobile_linear_velocity,
                                          -SystemContext.control_state.mobile_angular_velocity)
                            .set_minimum_time(Settings.dt * 1.01)
                        )
                        .set_body_command(
                            ctrl_builder
                        )
                    )
                )
            except Exception as e:
                logging.error(e)
                stream = None
                exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RB-Y1 Controller-Follow (UDP I/O)")
    parser.add_argument("--local_ip",   required=True, type=str, help="This PC bind IP for receiving controller commands")
    parser.add_argument("--local_port", required=True, type=int, help="UDP port to bind for controller commands")
    parser.add_argument("--control_ip", required=True, type=str, help="Controller PC IP to send robot/state")
    parser.add_argument("--control_port", required=True, type=int, help="Controller PC port to send robot/state")

    parser.add_argument("--no_gripper", action="store_true")
    parser.add_argument("--rby1", default="192.168.0.83:50051", type=str)
    parser.add_argument("--rby1_model", default="a", type=str)
    parser.add_argument("--no_head", action="store_true")
    parser.add_argument(
        "--whole_body", action="store_true",
        help="Use a whole-body optimization formulation (single control for all joints)"
    )


    args = parser.parse_args()
    main(args)
