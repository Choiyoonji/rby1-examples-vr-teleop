import json
import numpy as np
import matplotlib.pyplot as plt
from rby1_dyn import RBY1Dyn

def load_json_lines_file(file_path):
    """한 줄에 하나의 JSON 객체가 있는 파일을 읽어 리스트로 반환합니다."""
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # 빈 줄은 건너뛰기
                data_list.append(json.loads(line))
    # 만약 파일에 객체가 하나만 있었다면 리스트 대신 해당 객체를 반환하거나,
    # 여러 개가 있을 수 있다면 리스트 전체를 반환합니다.
    # 사용 방식에 맞게 선택하세요. 여기서는 리스트를 반환합니다.
    return data_list

def load_joint_trajectories(robot_data):
    joint_trajectories = []
    for frame in robot_data:
        if frame.get('is_torso_following', False):  # 토르소 팔로잉 상태인 프레임만 선택
            joint_trajectories.append({
                'timestamp': frame['timestamp'],
                'joint_positions': np.array(frame['joint_positions']),
                'right_target_ee': np.array(frame['right_target_ee']),
                'left_target_ee': np.array(frame['left_target_ee']),
            })
    return joint_trajectories

# def plot_joint_trajectory(joint_trajectories):
#     joint_num = len(joint_trajectories[0]['joint_positions'])
#     plt.figure(figsize=(12, 6))
#     for joint_index in range(2, joint_num-2):
#         timestamps = [frame['timestamp'] for frame in joint_trajectories]
#         joint_positions = [frame['joint_positions'][joint_index] for frame in joint_trajectories]

#         plt.subplot(joint_num, 1, joint_index)
#         plt.plot(timestamps[1000:], joint_positions[1000:], label=f'Joint {joint_index} Position')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Position (rad)')
#         plt.title(f'Joint {joint_index} Trajectory')
#         plt.legend()
#         plt.grid()
#     plt.tight_layout()
#     plt.show()

def plot_joint_trajectory(joint_trajectories, joint_idx):
    timestamps = [frame['timestamp'] for frame in joint_trajectories]
    joint_positions = [frame['joint_positions'][joint_idx] for frame in joint_trajectories]

    plt.plot(timestamps[1000:], joint_positions[1000:], label=f'Joint {joint_idx} Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.title(f'Joint {joint_idx} Trajectory')
    plt.legend()
    plt.grid()
    plt.show()

def plot_right_arm_joint_trajectory(joint_trajectories):
    timestamps = [frame['timestamp'] for frame in joint_trajectories]
    for joint_idx in range(8, 15):  # 오른팔 관절 인덱스 범위
        joint_positions = [frame['joint_positions'][joint_idx] for frame in joint_trajectories]

        plt.plot(timestamps[:], joint_positions[:], label=f'Joint {joint_idx} Position')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (rad)')
        plt.title(f'Joint {joint_idx} Trajectory')
        plt.legend()
        plt.grid()
    plt.show()

def plot_ee_pos_trajectory(joint_trajectories, rby1_dyn):
    timestamps = [frame['timestamp'] for frame in joint_trajectories]
    ee_positions = {'x': [], 'y': [], 'z': [], 'x_t': [], 'y_t': [], 'z_t': []}

    for frame in joint_trajectories:
        joint_positions = frame['joint_positions']
        fk_results = rby1_dyn.get_fk(joint_positions)
        left_ee_transform = fk_results['link_left_arm_6']
        ee_pos = left_ee_transform[:3, 3]  # 위치 추출
        ee_positions['x'].append(ee_pos[0])
        ee_positions['y'].append(ee_pos[1])
        ee_positions['z'].append(ee_pos[2])
        ee_positions['x_t'].append(frame['left_target_ee'][0, 3])
        ee_positions['y_t'].append(frame['left_target_ee'][1, 3])
        ee_positions['z_t'].append(frame['left_target_ee'][2, 3])

    plt.figure(figsize=(12, 8))
    for axis in ['x', 'y', 'z', 'x_t', 'y_t', 'z_t']:
        plt.plot(timestamps[:], ee_positions[axis][:], label=f'EE {axis.upper()} Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Right End-Effector Position Trajectory')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    rby1_dyn = RBY1Dyn()
    file_path = "/home/choiyj/rby1-examples-vr-teleop/python/logs/robot_state_20251020_191637.jsonl"  # 경로를 실제 파일 경로로 변경
    robot_data = load_json_lines_file(file_path)

    joint_trajectories = load_joint_trajectories(robot_data)

    # joint_index =  9  # 플롯할 관절 인덱스 설정
    # plot_joint_trajectory(joint_trajectories, joint_index)
    # plot_right_arm_joint_trajectory(joint_trajectories)
    plot_ee_pos_trajectory(joint_trajectories, rby1_dyn)