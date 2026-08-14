# -*- coding: utf-8 -*-
"""
estun_test.py — TCP 服务器,持续发送机械臂位姿 (xyzabc) 给 ESTUN 机器人,
并处理抓取/释放指令 (单向握手机制)

通信流程:
  1. 服务器持续发送位姿 [x,y,z,a,b,c]
  2. 机器人到达抓取位后发送 'GRIP'
     → 服务端 grip_object() 占位 → 回复 '111'
     → 继续发送位姿
  3. 机器人到达放置位后发送 'RELEASE'
     → 服务端 release_object() 占位 → 回复 '222'
     → 关闭连接

数据格式:
  发送: [x,y,z,a,b,c]  起始 '[',终止 ']',分隔符 ','
  接收: 纯字符串 GRIP / RELEASE (无分隔符)

机器人端调用示例:
  SocketCreate(sock, "192.168.101.96", 8080, ret)
  SocketReadStr(sock, str, 1000, ret, "[", "]")    ← 持续读位姿
  (到达抓取位后)
  SocketSendStr(sock, "GRIP", ret, "", "")
  SocketReadStr(sock, str, 1000, ret, "", "")       ← 读 '111'
  (到达放置位后)
  SocketSendStr(sock, "RELEASE", ret, "", "")
  SocketReadStr(sock, str, 1000, ret, "", "")        ← 读 '222'

用法:
  python estun_test.py
"""

import socket
import time

# ===== 服务器配置 =====
HOST = "192.168.101.96"  # 服务器 IP 地址(本机当前 IP)
PORT = 8080
SEND_INTERVAL = 0.1  # 发送间隔 (秒),≈ 10Hz

# ===== 模拟位姿数据 (可替换为 RealSense 实时数据) =====
pos_x = 0.477
pos_y = 0.231
pos_z = 0.624
angle_a = -135
angle_b = 4.5
angle_c = -86


# ===== 占位函数 =====

def grip_object():
    """
    抓取物体 (占位函数,后续可替换为实际抓取逻辑)
    """
    print("[ACTION] ===== 抓取物体 =====")
    time.sleep(0.5)  # 模拟抓取耗时


def release_object():
    """
    释放物体 (占位函数,后续可替换为实际释放逻辑)
    """
    print("[ACTION] ===== 释放物体 =====")
    time.sleep(0.5)  # 模拟释放耗时


# ===== 协议函数 =====

def build_pose_message(x, y, z, a, b, c):
    """
    构建位姿协议字符串:
    [x,y,z,a,b,c]
    xyz 保留3位小数,abc 保留2位小数
    """
    return f"[{x:.3f},{y:.3f},{z:.3f},{a:.2f},{b:.2f},{c:.2f}]"


def handle_client(conn, addr):
    """
    处理单个机器人连接:持续发送位姿 + 处理 GRIP/RELEASE 指令
    成功释放后关闭连接.
    """
    count = 0
    print(f"[INFO] 机器人已连接: {addr[0]}:{addr[1]}")
    conn.settimeout(0.05)  # 短超时,不阻塞发送循环

    try:
        while True:
            # ── 发送当前位姿 ──
            msg = build_pose_message(pos_x, pos_y, pos_z, angle_a, angle_b, angle_c)
            conn.sendall(msg.encode("utf-8"))
            count += 1

            if count % 10 == 0:
                print(f"[INFO] 已发送 {count} 帧: {msg}")

            # ── 非阻塞接收机械臂指令 ──
            try:
                data = conn.recv(1024)
                if not data:
                    print(f"[WARN] 机器人连接已关闭 (收到空数据)")
                    break
            except socket.timeout:
                # 没有数据,继续发送循环
                pass
            else:
                # 收到数据,解码并处理
                cmd = data.decode("utf-8").strip()
                print(f"[RECV] 收到指令: {repr(cmd)}")

                if cmd == "GRIP":
                    grip_object()
                    conn.sendall(b"GRIP_DONE")
                    print(f"[SEND] 回复: GRIP_DONE")

                elif cmd == "RELEASE":
                    release_object()
                    conn.sendall(b"RELEASE_DONE")
                    print(f"[SEND] 回复: RELEASE_DONE")
                    # 释放完成后关闭连接
                    break

                else:
                    print(f"[WARN] 未知指令: {cmd}")

            time.sleep(SEND_INTERVAL)

    except BrokenPipeError:
        print(f"[WARN] 机器人连接断开: {addr[0]}:{addr[1]}")
    except ConnectionResetError:
        print(f"[WARN] 连接被机器人重置: {addr[0]}:{addr[1]}")
    except KeyboardInterrupt:
        print(f"\n[INFO] 用户中断")
    except Exception as e:
        print(f"[ERROR] 异常: {e}")
    finally:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        conn.close()
        print(f"[INFO] 客户端连接已关闭: {addr[0]}:{addr[1]}")


def start_server():
    """启动 TCP 服务器,单连接处理"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    server.settimeout(1.0)
    print(f"[INFO] TCP 服务器已启动: {HOST}:{PORT}")
    print(f"[INFO] 等待机器人连接...")
    print(f"[INFO] 机器人端请调用:")
    print(f"      SocketCreate(sock, \"192.168.101.96\", 8080, ret)")
    print(f"[INFO] 按 Ctrl+C 退出服务器\n")

    try:
        while True:
            conn = None
            addr = None
            try:
                conn, addr = server.accept()
            except socket.timeout:
                pass

            if conn is not None:
                # 单连接处理:处理完 (释放完成或断开)后再接受下一个
                handle_client(conn, addr)
                print(f"[INFO] 单连接处理完毕,等待下一个机器人连接...\n")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断,服务器关闭")
    except Exception as e:
        print(f"[ERROR] 服务器异常: {e}")
    finally:
        server.close()
        print("[INFO] 服务器已关闭")


if __name__ == "__main__":
    start_server()