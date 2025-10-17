# command.py — waits for is_initialized before sending move frames
import json
import socket
import threading
import time
import cv2
import numpy as np
from typing import Dict, Any, List

# =======================
# 상수 설정
# =======================
TARGET_IP   = "127.0.0.1"   # follow_record.py의 local_ip
TARGET_PORT = 6001          # follow_record.py의 local_port
STATE_PORT  = 6002          # follow_record.py의 control_port

VR_JSON: str = "./vr_data.json"
SEND_DT: float = 0.05              # 20Hz
LOOP_PLAY: bool = False            # 반복 여부

# =======================
# 전역 상태 변수
# =======================
robot_state = {"is_initialized": False}


# =======================
# 헬퍼
# =======================
def build_command_from_record(frame: Dict[str, Any]) -> Dict[str, Any]:
    right = frame["hands"]["right"]
    left = frame["hands"]["left"]
    head = frame["head"]

    right_grip = 0.0 if right.get("buttons", {}).get("grip", 0.0) else 1.0
    left_grip  = 0.0 if left.get("buttons", {}).get("grip", 0.0) else 1.0

    cmd = {
        "arms": {
            "right": {
                "position": right["position"],
                "rotation": right["rotation"],
                "gripper": float(right_grip),
            },
            "left": {
                "position": left["position"],
                "rotation": left["rotation"],
                "gripper": float(left_grip),
            },
        },
        "head": {
            "position": head["position"],
            "rotation": head["rotation"],
        },
        "ready": False,
        "move": False,
        "stop": False,
        "estop": False,
    }
    return cmd


def send_udp_json(sock: socket.socket, target: tuple, payload: Dict[str, Any]) -> None:
    blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sock.sendto(blob, target)


# =======================
# 로봇 상태 수신 스레드
# =======================
def listen_robot_state():
    global robot_state
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", STATE_PORT))
    print(f"[INFO] Listening robot state on UDP {STATE_PORT}")
    while True:
        try:
            data, _ = sock.recvfrom(65535)
            msg = json.loads(data.decode("utf-8", errors="ignore"))
            if isinstance(msg, dict):
                robot_state.update(msg)
        except Exception as e:
            print("[WARN] State recv error:", e)


# =======================
# 메인 루프
# =======================
def main():
    # VR 데이터 로드
    with open(VR_JSON, "r") as f:
        frames: List[Dict[str, Any]] = json.load(f)
    if not frames:
        print("[ERROR] No frames in VR_JSON.")
        return

    target = (TARGET_IP, TARGET_PORT)
    print(f"[INFO] Target: {target}, 20 Hz fixed")
    print("[INFO] Controls: r=ready, m=move(toggle), s=stop, e=estop, q/ESC=quit")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 상태 수신 스레드 시작
    threading.Thread(target=listen_robot_state, daemon=True).start()

    # 키보드 플래그
    move_on = False
    one_shot_ready = False
    one_shot_stop = False
    one_shot_estop = False
    initialized = False

    # 빈 창 띄워 키 입력 가능하게 함
    cv2.namedWindow("command", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("command", 320, 80)

    idx = 0
    try:
        while True:
            frame = frames[idx]
            cmd = build_command_from_record(frame)

            # 키 입력 감지
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[INFO] Quit requested.")
                break
            elif key == ord("m"):
                move_on = not move_on
                print(f"[INFO] move toggled -> {move_on}")
            elif key == ord("r"):
                one_shot_ready = True
                print("[INFO] ready pulse (move disabled until initialized)")
                move_on = False
            elif key == ord("s"):
                one_shot_stop = True
                print("[INFO] stop pulse")
                move_on = False
            elif key == ord("e"):
                one_shot_estop = True
                print("[WARN] E-STOP pulse")
                move_on = False

            # 로봇 상태 확인
            initialized = bool(robot_state.get("is_initialized", False))

            # ready, stop, estop 우선 적용
            cmd["ready"] = one_shot_ready
            cmd["stop"] = one_shot_stop
            cmd["estop"] = one_shot_estop

            # move는 is_initialized가 True일 때만 가능
            cmd["move"] = move_on and initialized

            # UDP 전송
            send_udp_json(sock, target, cmd)

            # ready 전송 후 초기화 완료될 때까지 대기
            if one_shot_ready:
                print("[INFO] Waiting for robot initialization...")
                while not robot_state.get("is_initialized", False):
                    time.sleep(0.1)
                print("[INFO] Robot initialized! Move enabled.")
                # move_on = True
                one_shot_ready = False

            # 펄스 플래그는 매 프레임 클리어
            one_shot_stop = False
            one_shot_estop = False

            # move 활성화 상태에서만 프레임 진행
            if move_on and initialized:
                idx += 1
                if idx >= len(frames):
                    if LOOP_PLAY:
                        idx = 0
                    else:
                        print("[INFO] Finished one pass.")
                        break

            # 20Hz 고정
            time.sleep(SEND_DT)

    finally:
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
