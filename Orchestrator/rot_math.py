"""姿态旋转数学工具 (仅依赖 numpy).

B2 动态抓取落地的数学内核: 四元数/旋转矩阵互转、旋转测地距离、
四元数均值 (Markley)、帧间 4 候选锚定 (align_axes)、矩阵 EMA 平滑、
euler 解缠、Rodrigues (numpy 纯实现)、位姿单位纪律检查.

迁移自 Orchestrator/handle_cup_detector.py (HEAD 7c7a6fd, 传统线实现):
  - rotate_around_axis / align_axes / MatrixSmoother / unwrap_euler
  (公开命名; 检测器侧改为 import 复用, 单份维护)

姿态连续性的语义约定 (审计 B2): 内部一律以旋转矩阵/四元数传递与判定
(旋转测地距离), 仅在变换到基座系后才生成 ESTUN ABC (euler).
"""

import math

import numpy as np


# ============================================================
#  四元数
# ============================================================

def rotation_matrix_to_quaternion(R):
    """旋转矩阵 → 四元数 (w,x,y,z), Shepperd 法, w>=0.

    R 非正交时先 SVD 正交化 (U@Vt); 行列式为负时翻 U 第三列.
    """
    R = np.asarray(R, dtype=float)
    U, _, Vt = np.linalg.svd(R)
    Rq = U @ Vt
    if np.linalg.det(Rq) < 0.0:
        U[:, 2] *= -1.0
        Rq = U @ Vt
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = Rq.ravel()
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (m21 - m12) / s,
                      (m02 - m20) / s,
                      (m10 - m01) / s])
    elif m00 >= m11 and m00 >= m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = np.array([(m21 - m12) / s,
                      0.25 * s,
                      (m01 + m10) / s,
                      (m02 + m20) / s])
    elif m11 >= m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = np.array([(m02 - m20) / s,
                      (m01 + m10) / s,
                      0.25 * s,
                      (m12 + m21) / s])
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = np.array([(m10 - m01) / s,
                      (m02 + m20) / s,
                      (m12 + m21) / s,
                      0.25 * s])
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def quaternion_to_rotation_matrix(q):
    """四元数 (w,x,y,z) → 旋转矩阵 (正交)."""
    w, x, y, z = np.asarray(q, dtype=float)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


# ============================================================
#  测地距离与均值
# ============================================================

def rotation_geodesic_deg(R1, R2):
    """两旋转矩阵的测地距离 (度) = 相对旋转角 = arccos((tr(R1ᵀR2)−1)/2).

    对 ±180° 等价表示与 PCA 轴符号翻转不敏感 (审计 B2: 用旋转测地距离
    而非欧拉角逐分量差做连续性判断). 与 quaternion_geodesic_deg 数值一致
    (四元数 S³ 距离 2·arccos(|q1·q2|) 恰为旋转角 θ 的两倍一半).
    """
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    return math.degrees(math.acos(c))


def quaternion_geodesic_deg(q1, q2):
    """两单位四元数的测地距离 (度), 处理 double cover (|q1·q2|)."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    c = min(1.0, max(-1.0, float(abs(np.dot(q1, q2)))))
    return math.degrees(2.0 * math.acos(c))


def quaternion_mean(quats):
    """四元数均值 (Markley 特征向量法): A = Σ qᵢqᵢᵀ 最大特征向量.

    quats: (N,4) 行向量 (w,x,y,z), 自动归一化; double cover 下良定义
    (q 与 -q 同一旋转, 特征向量法自动消歧).
    """
    Q = np.asarray(quats, dtype=float).reshape(-1, 4)
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    Q = Q / np.maximum(norms, 1e-12)
    A = Q.T @ Q
    w, v = np.linalg.eigh(A)
    q = v[:, int(np.argmax(w))]
    if q[0] < 0.0:
        q = -q
    return q / np.linalg.norm(q)


# ============================================================
#  帧间锚定 / 平滑 / 解缠 (迁移自 handle_cup_detector, 7c7a6fd)
# ============================================================

def rotate_around_axis(axis, angle_deg):
    """绕任意轴 axis 旋转 angle_deg 的旋转矩阵 (numpy 纯 Rodrigues).

    返回 3x3 旋转矩阵 (作用于列向量, 即 R @ v). 与 cv2.Rodrigues 一致
    (零模轴返回单位阵).
    """
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-10:
        return np.eye(3)
    axis = axis / n
    theta = math.radians(angle_deg)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def align_axes(rot, prev_rot):
    """帧间姿态锚定: 在 4 个候选 (恒等 + 绕 X/Y/Z 各 180°) 中选与上一帧最接近者.

    治 PCA 主轴符号逐帧任意导致的帧间 180° 翻转 (根因 1).
    覆盖全部翻转模式:
      - Y 翻转 (绕壶身轴 X 180°)
      - Z 翻转 (绕把手轴 Y 180°, Z[2]≈0 时 z 符号约束失效的兜底)
      - X/Y 组合翻转 (绕相机 Z 180°)
    锚点: 上一帧旋转矩阵 (帧间锚定, 不依赖正对参考).
    """
    if prev_rot is None:
        return np.asarray(rot, dtype=float)
    rot = np.asarray(rot, dtype=float)
    cands = [rot]
    for axis_i in range(3):
        cands.append(rotate_around_axis(rot[:, axis_i], 180.0) @ rot)
    return max(cands, key=lambda R: float(np.trace(R.T @ prev_rot)))


class MatrixSmoother:
    """旋转矩阵 EMA 平滑 + SVD 正交化.

    在矩阵空间平滑, 避开 euler 万向锁区域的帧间跳变 (根因 2 放大因子):
    接近 B≈±90° 时 euler 提取对微小矩阵噪声敏感 ~100 倍, 矩阵 EMA 使
    输出平滑摆动而非跳变.
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.filtered = None

    def update(self, rot):
        rot = np.asarray(rot, dtype=float)
        if self.filtered is None:
            self.filtered = rot.copy()
            return self.filtered
        self.filtered = (1.0 - self.alpha) * self.filtered + self.alpha * rot
        U, _, Vt = np.linalg.svd(self.filtered)
        self.filtered = U @ Vt
        return self.filtered

    def reset(self):
        self.filtered = None


def unwrap_euler(euler_new, euler_prev):
    """A/C 取与上一帧最接近的 ±360k 等价表示 (治 ±180 边界表观跳变).

    B 由矩阵 EMA 保证平滑, 不需要 wrap.
    """
    out = list(euler_new)
    for i in (0, 2):
        while out[i] - euler_prev[i] > 180:
            out[i] -= 360
        while euler_prev[i] - out[i] > 180:
            out[i] += 360
    return tuple(out)


# ============================================================
#  单位纪律 (B1)
# ============================================================

def sanity_check_pose(values6, unit):
    """位姿单位纪律: 解析边界显式声明单位, 越界抛 ValueError.

    协议双约定: handle 相位 (8080) = m/deg, greeting (8600) = mm/deg.
    - unit="m":  |xyz| ≤ 5 m, |abc| ≤ 720°
    - unit="mm": |xyz| ≤ 5000 mm, |abc| ≤ 720°
    调用点: state_machine Phase A 解析 (m), greeting _handle_session (mm).
    """
    vals = np.asarray(values6, dtype=float).reshape(-1)
    if vals.shape != (6,):
        raise ValueError(f"sanity_check_pose 需要 6 个数值, 得到 {vals.shape}")
    if unit not in ("m", "mm"):
        raise ValueError(f"未知单位: {unit!r} (仅支持 m/mm)")
    limit_xyz = 5.0 if unit == "m" else 5000.0
    if np.any(np.abs(vals[:3]) > limit_xyz):
        raise ValueError(f"位姿 {unit} 越界: xyz={vals[:3].tolist()} (限 ±{limit_xyz}{unit})")
    if np.any(np.abs(vals[3:]) > 720.0):
        raise ValueError(f"位姿角度越界: abc={vals[3:].tolist()} (限 ±720°)")
