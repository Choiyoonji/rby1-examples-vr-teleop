
# command.py — waits for is_initialized before sending move frames
import json
import socket
import threading
import time
import cv2
import numpy as np
from typing import Dict, Any, List
from collections import deque
import os

# =======================
# 상수 설정
# =======================
TARGET_IP   = "127.0.0.1"   # follow_record.py의 local_ip
TARGET_PORT = 6001          # follow_record.py의 local_port
STATE_PORT  = 6002          # follow_record.py의 control_port

VR_JSON: str = "./vr_data_10hz.json"
SEND_DT: float = 0.1              # 10Hz
LOOP_PLAY: bool = False            # 반복 여부

# ---- robot_state 저장 관련 ----
# 세션별 파일명(실행 시각 기반)으로 JSON Lines(ndjson) 포맷으로 저장
STATE_LOG_DIR = "./logs"
os.makedirs(STATE_LOG_DIR, exist_ok=True)
STATE_LOG_PATH = os.path.join(STATE_LOG_DIR, time.strftime("robot_state_%Y%m%d_%H%M%S.jsonl"))
STATE_WRITE_INTERVAL = 0.5   # 버퍼 flush 주기(초)
STATE_BUFFER_MAXLEN = 5000   # 버퍼 최대 길이(초과 시 가장 오래된 항목 제거)

# =======================
# 전역 상태 변수
# =======================
robot_state = {"is_initialized": False}
_state_buffer = deque(maxlen=STATE_BUFFER_MAXLEN)
_state_writer_running = True


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
    """follow_record.py에서 주는 robot_state(UDP)를 수신하여
    - 최신 상태를 robot_state(dict)에 반영
    - 디스크로 저장할 수 있도록 버퍼에 적재
    JSON Lines(.jsonl) 포맷: 메시지 1개당 한 줄
    """
    global robot_state
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", STATE_PORT))
    print(f"[INFO] Listening robot state on UDP {STATE_PORT}")
    while True:
        try:
            data, _ = sock.recvfrom(65535)
            msg = json.loads(data.decode("utf-8", errors="ignore"))
            if isinstance(msg, dict):
                # 최신 상태 업데이트
                robot_state.update(msg)
                # 수신 시각 태그 추가(로컬 시간)
                msg_with_ts = {"_recv_time": time.time(), **msg}
                _state_buffer.append(msg_with_ts)
        except Exception as e:
            print("[WARN] State recv error:", e)


def _state_writer_loop():
    """버퍼에 쌓인 robot_state를 주기적으로 파일(STATE_LOG_PATH)에 기록"""
    print(f"[INFO] Robot state logging to: {STATE_LOG_PATH}")
    with open(STATE_LOG_PATH, "a", encoding="utf-8") as f:
        while _state_writer_running:
            try:
                # 버퍼 비우기
                batch = []
                while _state_buffer:
                    batch.append(_state_buffer.popleft())
                if batch:
                    for item in batch:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                time.sleep(STATE_WRITE_INTERVAL)
            except Exception as e:
                print("[WARN] State writer error:", e)
                time.sleep(STATE_WRITE_INTERVAL)


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
    # 상태 파일 기록 스레드 시작
    threading.Thread(target=_state_writer_loop, daemon=True).start()

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
                    print("waiting for initialize")
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
                print(idx)
                if idx >= len(frames):
                    if LOOP_PLAY:
                        idx = 0
                    else:
                        print("[INFO] Finished one pass.")
                        break

            # 20Hz 고정
            time.sleep(SEND_DT)

    finally:
        # 종료 시 writer loop도 종료되도록 플래그 변경
        global _state_writer_running
        _state_writer_running = False
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()