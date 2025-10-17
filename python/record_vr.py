import argparse
import logging
import time
import threading
from dataclasses import dataclass
import socket
from typing import Union
import json
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from vr_control_state import VRControlState


class SystemContext:
    vr_state = VRControlState()


def main():
    local_ip = "192.168.0.221"
    local_port = 5005
    meta_quest_ip = "192.168.0.106"
    meta_quest_port = 6000
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        target_info = {
            "ip": local_ip,
            "port": local_port
        }
        message = json.dumps(target_info).encode('utf-8')
        sock.sendto(message, (meta_quest_ip, meta_quest_port))
        logging.info(f"Sent local PC info to Meta Quest: {target_info}")

    data_path = "vr_data.json"

    data_list = []

    dt = 0.1  # 10 Hz
    last_time = time.monotonic()
    next_time = last_time + dt
    is_initialized = False

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server_sock:
        server_sock.bind((local_ip, local_port))
        logging.info(f"UDP server running to receive Meta Quest Controller data... {local_ip}:{local_port}")
        while True:
            data, addr = server_sock.recvfrom(4096)
            udp_msg = data.decode('utf-8')
            try:
                SystemContext.vr_state.controller_state = json.loads(udp_msg)
                rate = 1.0 / (time.monotonic() - last_time)
                last_time = time.monotonic()

                print(f"Received data from {addr} at {rate:.1f} Hz", end='\r')

                if not is_initialized and SystemContext.vr_state.controller_state["hands"]["right"]["buttons"]["primaryButton"]:
                    is_initialized = True
                    logging.info("VR Control State Initialized")

                if is_initialized:
                    if time.monotonic() > next_time:
                        data_list.append(SystemContext.vr_state.controller_state)
                        next_time = time.monotonic() + dt
                    if SystemContext.vr_state.controller_state["hands"]["right"]["buttons"]["secondaryButton"]:
                        with open(data_path, 'w') as f:
                            json.dump(data_list, f, indent=4)
                        logging.info(f"Data saved to {data_path}, total frames: {len(data_list)}")
                        break

            except json.JSONDecodeError as e:
                logging.warning(f"Failed to decode JSON: {e} (from {addr}) - received data: {message[:100]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()