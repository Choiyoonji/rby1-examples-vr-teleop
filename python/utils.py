import numpy as np
from scipy.spatial.transform import Rotation as R

# 1) Euler → Quaternion
def euler_to_quat(euler, seq='xyz', degrees=True):
    """
    euler: array-like, shape (3,)  [roll, pitch, yaw] (seq에 맞춤)
    return: np.ndarray, shape (4,) quaternion [x, y, z, w]
    """
    euler = np.asarray(euler, dtype=float)
    return R.from_euler(seq, euler, degrees=degrees).as_quat()

# 2) Quaternion → Euler
def quat_to_euler(quat, seq='xyz', degrees=True):
    """
    quat: array-like, shape (4,)  quaternion [x, y, z, w]
    return: np.ndarray, shape (3,) Euler angles in seq order
    """
    quat = np.asarray(quat, dtype=float)
    return R.from_quat(quat).as_euler(seq, degrees=degrees)

# 3) SE(3) → position
def se3_to_pos(T):
    """
    T: np.ndarray, shape (4,4)  homogeneous transform
    return: np.ndarray, shape (3,) position
    """
    T = np.asarray(T, dtype=float)
    assert T.shape == (4, 4), "T must be 4x4"
    return T[:3, 3].copy()

# 4) position + quaternion → SE(3)
def pos_to_se3(position, rotation_quat):
    """
    position: array-like, shape (3,)
    rotation_quat: array-like, shape (4,) quaternion [x, y, z, w]
    return: np.ndarray, shape (4,4) SE(3)
    """
    position = np.asarray(position, dtype=float)
    rotation_quat = np.asarray(rotation_quat, dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_quat(rotation_quat).as_matrix()
    T[:3, 3]  = position
    return T