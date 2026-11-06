# -*- coding: utf-8 -*-
"""
mold_synth.py
-------------
奶酪表面污染霉菌的物理启发式合成 + 工业相机成像链路仿真。

霉菌外观依据（文献/权威科普归纳）：
  * 污染型霉菌（非发酵剂）以簇状菌落出现，形态为 *绒毛状 / 粉末状*；
  * 颜色谱：灰白、橄榄绿、蓝绿（青霉 Penicillium commune 类）、
    墨绿/黑（曲霉 Aspergillus 类）、偶见粉红/红褐；
  * 菌落中心致密、颜色深，边缘呈白色绒毛状晕圈并向外扩散成不规则轮廓；
  * 菌落周围常伴"潮湿晕"——表皮发暗、失水、轻微变色；
  * 偏好生长在表皮褶皱、划痕、低洼处（局部纹理能量高的区域）。

合成策略：
  1) 生成"奶酪区"掩膜（避免把霉斑撒到背景/木板上）；
  2) 以局部纹理方差为权重采样菌落中心，聚成 1~3 个菌落簇；
  3) 每个菌落用角度域随机半径场生成不规则边界 + 多尺度 fbm 噪声做绒毛边缘；
  4) 用"高度场-朗伯"近似给菌落加上体积感（边缘受光变亮 = 绒毛感）；
  5) 叠加潮湿晕；最后统一走相机成像链路（光照梯度、暗角、色温、离焦、
     运动模糊、泊松散粒噪声、读出噪声、JPEG 压缩）。
"""

import numpy as np
import cv2

# ---------------------------------------------------------------- 噪声工具

def _value_noise(h, w, cells, rng):
    """低分辨率随机场 → 双三次上采样，得到平滑值噪声。"""
    gw = max(3, int(cells) + 2)
    gh = max(3, int(round(gw * h / max(w, 1))) + 1)
    g = rng.random((gh, gw)).astype(np.float32)
    out = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(out, 0.0, 1.0)


def fbm(h, w, rng, octaves=4, cells=3):
    """分形布朗噪声，归一化到 [0,1]。"""
    total = np.zeros((h, w), np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        total += amp * _value_noise(h, w, cells * (2 ** o), rng)
        norm += amp
        amp *= 0.5
    out = total / max(norm, 1e-6)
    out -= out.min()
    out /= (out.max() + 1e-6)
    return out


def _periodic_smooth(x, half=3):
    """环形（周期）平滑：用于角度域轮廓，保证 0 与 2π 处连续。"""
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / (half / 1.6)) ** 2).astype(np.float32)
    k /= k.sum()
    ext = np.concatenate([x[-half:], x, x[:half]])
    sm = np.convolve(ext, k, mode="same")
    return sm[half:half + len(x)].astype(np.float32)


# ---------------------------------------------------------------- 霉菌色板

# (core_rgb, edge_rgb, name)  —— 核心色深、边缘色浅（绒毛反光）
MOLD_SPECIES = [
    ((210, 210, 198), (238, 238, 231), "灰白绒霉 Penicillium(white)"),
    ((86, 112, 74),   (152, 172, 128), "橄榄绿青霉 Penicillium(olive)"),
    ((60, 98, 106),   (130, 162, 152), "蓝绿青霉 Penicillium(blue-green)"),
    ((40, 40, 38),    (98, 100, 94),   "黑曲霉 Aspergillus(niger)"),
    ((148, 70, 78),   (198, 142, 142), "粉红污染霉 Fusarium/yeast"),
]
SPECIES_WEIGHTS = np.array([0.30, 0.30, 0.20, 0.12, 0.08])


def _pick_species(rng):
    i = int(rng.choice(len(MOLD_SPECIES), p=SPECIES_WEIGHTS / SPECIES_WEIGHTS.sum()))
    core, edge, name = MOLD_SPECIES[i]
    # 每个样本再叠加一点个体色差
    jitter = rng.normal(0, 6, 3)
    core = np.clip(np.array(core, np.float32) + jitter, 0, 255)
    edge = np.clip(np.array(edge, np.float32) + jitter, 0, 255)
    return core.astype(np.float32), edge.astype(np.float32), name


# ---------------------------------------------------------------- 奶酪区掩膜

def cheese_mask(bgr, erode_px=0):
    """粗分割出奶酪本体（亮、低-中饱和、略偏黄），用于限制撒霉区域。"""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    Lf = cv2.GaussianBlur(L, (0, 0), max(1.0, min(h, w) / 120.0))
    thr, _ = cv2.threshold(Lf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (Lf > max(thr, 90)).astype(np.uint8) * 255
    # 再要求"偏黄"（Lab b* 高），排除白盘子、灰台面等亮而中性的区域
    bch = lab[:, :, 2]
    yellow = (bch > 133)
    # 再要求"局部平滑"：奶酪表面即便有纹理，其高频能量也远低于花岗岩台面/织物
    s1 = cv2.GaussianBlur(Lf, (0, 0), 2.0)
    s2 = cv2.GaussianBlur(Lf * Lf, (0, 0), 2.0)
    lstd = np.clip(s2 - s1 * s1, 0, None)
    smooth = (lstd < max(6.0, float(np.percentile(lstd, 60))))
    m_y = ((m > 0) & yellow & smooth).astype(np.uint8) * 255
    if m_y.mean() > 6.0:          # 至少 6% 面积才采用更严格的掩膜
        m = m_y

    k = max(3, int(min(h, w) * 0.02)) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)

    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab_cc == biggest).astype(np.uint8) * 255
    if erode_px > 0:
        k2 = int(erode_px) | 1
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))
    return m


# ---------------------------------------------------------------- 单菌落渲染

def _colony_layer(mask_shape, cx, cy, r, rng, core, edge):
    """返回 (patch_y0, patch_x0, alpha, colorRGB) ，越界自动裁剪。"""
    h, w = mask_shape
    pad = int(r * 2.4) + 6
    x0, x1 = max(0, int(cx - pad)), min(w, int(cx + pad))
    y0, y1 = max(0, int(cy - pad)), min(h, int(cy + pad))
    hh, ww = y1 - y0, x1 - x0
    if hh < 8 or ww < 8:
        return None

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    d = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)

    # 角度域周期性随机半径场 → 不规则轮廓
    nb = 96
    prof = rng.random(nb).astype(np.float32)
    prof = _periodic_smooth(prof, half=4)
    prof -= prof.min()
    prof /= (prof.max() + 1e-6)
    idx = np.clip(((ang + np.pi) / (2 * np.pi) * nb).astype(np.int32), 0, nb - 1)
    rfield = r * (0.70 + 0.50 * prof[idx])

    feather = max(1.5, r * 0.30)
    alpha = np.clip((rfield - d) / feather, 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)          # smoothstep

    # 绒毛 / 粉末质感：多尺度噪声调制密度
    n_edge = fbm(hh, ww, rng, octaves=4, cells=max(2, int(r / 5)))
    alpha = np.clip(alpha * (0.30 + 1.25 * n_edge), 0.0, 1.0)

    # 中心致密、外围稀薄
    t = np.clip(1.0 - d / np.maximum(rfield, 1e-3), 0.0, 1.0)[..., None]
    color = edge[None, None, :] * (1.0 - t ** 0.75) + core[None, None, :] * (t ** 0.75)

    # 细颗粒纹理
    tex = 0.80 + 0.40 * fbm(hh, ww, rng, octaves=3, cells=max(3, int(r / 2.5)))
    color = color * tex[..., None]

    # 高度场→朗伯近似：边缘隆起受光变亮，形成"绒毛受光"观感
    dome = cv2.GaussianBlur(alpha, (0, 0), max(1.0, r * 0.30))
    lam = np.clip(0.86 + 2.9 * (dome - alpha), 0.55, 1.32)
    color = color * lam[..., None]

    return (y0, y1, x0, x1, alpha.astype(np.float32),
            np.clip(color, 0, 255).astype(np.float32))


def apply_mold(bgr, rng, coverage=(0.02, 0.09), species=None):
    """
    在奶酪表面合成污染霉菌。
    coverage: 期望的菌落簇外接面积占比区间（相对于图像面积）。
    返回 (合成图, 元信息 dict)
    """
    h, w = bgr.shape[:2]
    if species is None:
        core, edge, sname = _pick_species(rng)
    else:
        core, edge, sname = species

    tgt_total = rng.uniform(*coverage) * h * w

    # 采样权重：奶酪区内（黄、饱和）+ 纹理能量高（褶皱/划痕/低洼）处更易长霉
    valid = cheese_mask(bgr, erode_px=int(min(h, w) * 0.06))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hc, Sc = hsv[:, :, 0], hsv[:, :, 1]
    yellowish = ((Hc >= 10) & (Hc <= 48) & (Sc >= 50)).astype(np.uint8) * 255
    v2 = ((valid > 0) & (yellowish > 0)).astype(np.uint8) * 255
    if v2.mean() > 3.0:
        valid = v2
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_std = cv2.GaussianBlur(L * L, (0, 0), 3.0) - cv2.GaussianBlur(L, (0, 0), 3.0) ** 2
    local_std = np.clip(local_std, 0, None)
    weight = (valid > 0) * (0.25 + local_std / (local_std.max() + 1e-6))
    weight_sum = weight.sum()
    if weight_sum < 1e-3:
        return bgr, {"species": sname, "colonies": 0, "coverage": 0.0}

    flat = weight.ravel() / weight.sum()
    r_base = max(6.0, min(h, w) * rng.uniform(0.030, 0.070))
    acc_a = np.zeros((h, w), np.float32)
    acc_c = np.zeros((h, w, 3), np.float32)

    def _stamp(cx, cy, r):
        lay = _colony_layer((h, w), cx, cy, r, rng, core, edge)
        if lay is None:
            return 0.0
        y0, y1, x0, x1, a, c = lay
        sub_a = acc_a[y0:y1, x0:x1]
        sub_c = acc_c[y0:y1, x0:x1]
        add = a * (0.62 + 0.33 * rng.random())        # 不透明度：霉是半透明覆盖
        m = add > sub_a
        sub_c[m] = c[m]
        sub_a[m] = add[m]
        return float(a.mean())

    n_clusters = int(rng.integers(1, 4))
    n_col = 0
    centers = rng.choice(h * w, size=n_clusters, replace=False, p=flat)
    gained = 0.0
    for c_id in centers:
        cy, cx = divmod(int(c_id), w)
        spread = r_base * rng.uniform(1.05, 2.45)
        n_here = int(rng.integers(9, 32))
        for _ in range(n_here):
            ox, oy = rng.normal(0, spread * 0.55, 2)
            dist = float(np.hypot(ox, oy))
            if dist > spread * 1.5:
                continue
            r = r_base * rng.uniform(0.45, 1.15) * (1.0 - 0.45 * min(dist / (spread + 1e-6), 1.0))
            if _stamp(cx + ox, cy + oy, max(3.0, r)):
                n_col += 1
                gained += np.pi * r * r
        # 菌落簇外的零散卫星斑点
        for _ in range(int(rng.integers(0, 5))):
            j = int(rng.choice(h * w, p=flat))
            sy, sx = divmod(j, w)
            if _stamp(sx, sy, max(2.5, r_base * rng.uniform(0.20, 0.45))):
                n_col += 1
                gained += np.pi * (r_base * 0.3) ** 2

    # 潮湿晕：菌落周围表皮发暗、轻微变色
    if acc_a.max() > 0:
        halo = cv2.GaussianBlur(acc_a, (0, 0), max(3.0, r_base * 1.4))
        halo = np.clip(halo * 0.55, 0, 0.55)[..., None]
        tint = np.array([200.0, 205.0, 190.0], np.float32)   # 微微发灰绿
        img_f = bgr.astype(np.float32)
        img_f = img_f * (1.0 - 0.22 * halo) + tint * (0.10 * halo)
    else:
        img_f = bgr.astype(np.float32)

    a3 = acc_a[..., None]
    out = img_f * (1.0 - a3) + acc_c * a3
    out = np.clip(out, 0, 255).astype(np.uint8)

    meta = {
        "species": sname,
        "colonies": int(n_col),
        "coverage": float(acc_a.mean()),
    }
    return out, meta


# ---------------------------------------------------------------- 相机成像链路

def camera_simulate(bgr, rng, strength=1.0):
    """把"理想图"退化成工业相机在产线上的真实成像：光照、色温、离焦、
    运动模糊、散粒噪声、读出噪声、JPEG 压缩。两类图片共用同一分布。"""
    h, w = bgr.shape[:2]
    out = bgr.astype(np.float32)

    # 1) 非均匀光照（灯位偏移）+ 暗角
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx = rng.uniform(-0.20, 0.20) * strength
    gy = rng.uniform(-0.14, 0.14) * strength
    ramp = 1.0 + gx * (xx / w - 0.5) * 2.0 + gy * (yy / h - 0.5) * 2.0
    r2 = (xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2
    vig = 1.0 - rng.uniform(0.0, 0.38) * strength * r2 * 2.4
    out *= (ramp * vig)[..., None]

    # 2) 色温漂移（BGR 顺序）
    out[..., 0] *= rng.uniform(0.93, 1.08)
    out[..., 1] *= rng.uniform(0.97, 1.03)
    out[..., 2] *= rng.uniform(0.93, 1.09)

    # 3) 离焦
    sigma = rng.uniform(0.0, 1.15) * strength
    if sigma > 0.08:
        out = cv2.GaussianBlur(out, (0, 0), sigma)

    # 4) 运动模糊（传送带/机械臂振动）
    if rng.random() < 0.25 * strength:
        k = int(rng.integers(3, 10))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
        ker = cv2.warpAffine(ker, M, (k, k))
        ker /= (ker.sum() + 1e-6)
        out = cv2.filter2D(out, -1, ker)

    out = np.clip(out, 0, 255)

    # 5) 伽马（相机 tone curve）
    out = 255.0 * np.power(out / 255.0, rng.uniform(0.90, 1.14))

    # 6) 散粒噪声（泊松）+ 读出噪声（高斯）
    gain = rng.uniform(0.7, 1.7)
    out = rng.poisson(np.clip(out, 1e-3, 255) * gain) / gain
    out = out + rng.normal(0, rng.uniform(1.2, 7.5) * strength, out.shape)

    # 7) 量化 + JPEG
    out = np.clip(out, 0, 255).astype(np.uint8)
    q = int(rng.integers(48, 94))
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out
