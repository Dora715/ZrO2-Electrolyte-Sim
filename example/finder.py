# -*- coding: utf-8 -*-
import numpy as np
import os
from ase.io import read
from ase import Atoms


def find_migration_target(pwo_file, ref_vector=np.array([1.0, 0.0, 0.0])):
    """
    根据黄金4步逻辑，自动寻找最佳跃迁氧原子和空位坐标。
    返回: (最佳氧原子的QE序号(从1开始), 空位坐标[x, y, z])
    """
    print(f"\n>>> 正在分析文件: {pwo_file}")

    # 读取弛豫后的结构
    try:
        atoms = read(pwo_file, format='espresso-out')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None, None

    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())

    cations = ['Zr', 'Sc', 'Y']
    o_indices = [i for i, sym in enumerate(symbols) if sym == 'O']
    cation_indices = [i for i, sym in enumerate(symbols) if sym in cations]

    # ==========================================
    # 1. 确定氧空位 (V_O) 的精确坐标
    # ==========================================
    undercoord_cations = []
    for c_idx in cation_indices:
        # 获取该阳离子到所有氧原子的距离（自动处理周期性边界）
        distances = atoms.get_distances(c_idx, o_indices, mic=True)
        coord_num = np.sum(distances <= 3.0)
        if coord_num == 7:
            undercoord_cations.append(c_idx)

    if len(undercoord_cations) != 4:
        print(f"警告: 找到了 {len(undercoord_cations)} 个 7 配位阳离子，期望值是 4！")
        if len(undercoord_cations) == 0:
            return None, None

    # 为了安全求跨边界的几何中心，以第一个阳离子为基准，利用 MIC 向量求平均
    ref_cat_idx = undercoord_cations[0]
    ref_pos = positions[ref_cat_idx]

    if len(undercoord_cations) > 1:
        other_cat_indices = undercoord_cations[1:]
        # 获取基准原子指向其他配位不足阳离子的向量
        vecs = atoms.get_distances(ref_cat_idx, other_cat_indices, mic=True, vector=True)
        # 空位坐标 = 基准原子坐标 + 所有向量的平均偏移量（包括它自己也就是0向量）
        mean_offset = np.sum(vecs, axis=0) / len(undercoord_cations)
        vac_pos = ref_pos + mean_offset
    else:
        vac_pos = ref_pos

    print(f"[*] 成功定位氧空位坐标: {vac_pos.round(4)}")

    # 往系统中临时插入一个虚拟原子（如 Helium）代表空位，方便后续调用 ASE 计算距离
    temp_atoms = atoms.copy()
    temp_atoms.append('He')
    vac_idx = len(temp_atoms) - 1
    temp_atoms.positions[vac_idx] = vac_pos

    # ==========================================
    # 2. 扫描并提取候选氧原子 (O_candidates)
    # ==========================================
    vecs_to_O = temp_atoms.get_distances(vac_idx, o_indices, mic=True, vector=True)
    dists_to_O = np.linalg.norm(vecs_to_O, axis=1)

    candidate_indices = []
    candidate_vectors = []

    for i, dist in enumerate(dists_to_O):
        if dist <= 2.8:
            candidate_indices.append(o_indices[i])
            candidate_vectors.append(vecs_to_O[i])  # 空位指向该 O 原子的向量

    print(f"[*] 在 2.8 Å 范围内找到 {len(candidate_indices)} 个候选氧原子。")

    # ==========================================
    # 3. 判定跃迁瓶颈元素 (核心淘汰机制)
    # ==========================================
    valid_candidates = []
    valid_vectors = []

    for c_idx, c_vec in zip(candidate_indices, candidate_vectors):
        # 跃迁中点坐标 = 空位坐标 + 1/2向量
        midpoint_pos = vac_pos + c_vec / 2.0

        # 临时将该中点设为虚拟原子
        temp_mid = atoms.copy()
        temp_mid.append('He')
        mid_idx = len(temp_mid) - 1
        temp_mid.positions[mid_idx] = midpoint_pos

        # 寻找距离中点最近的阳离子
        distances_to_cats = temp_mid.get_distances(mid_idx, cation_indices, mic=True)
        nearest_2_indices = np.argsort(distances_to_cats)[:2]

        cat1 = symbols[cation_indices[nearest_2_indices[0]]]
        cat2 = symbols[cation_indices[nearest_2_indices[1]]]

        if cat1 == 'Zr' and cat2 == 'Zr':
            valid_candidates.append(c_idx)
            valid_vectors.append(c_vec)

    print(f"[*] 经过 Zr-Zr 瓶颈筛选，剩余 {len(valid_candidates)} 个合法候选者。")

    # ==========================================
    # 4. 空间方向一致性 (终极选择机制)
    # ==========================================
    if not valid_candidates:
        print("[!] 错误: 没有氧原子满足纯 Zr-Zr 瓶颈！请检查结构是否畸变过大。")
        return None, vac_pos.tolist()

    ref_vector = ref_vector / np.linalg.norm(ref_vector)
    min_angle = float('inf')
    best_o_idx = None

    for c_idx, c_vec in zip(valid_candidates, valid_vectors):
        norm_c_vec = c_vec / np.linalg.norm(c_vec)
        # 点乘求夹角，限制范围防止浮点数越界
        dot_product = np.clip(np.dot(norm_c_vec, ref_vector), -1.0, 1.0)
        angle = np.arccos(dot_product)

        if angle < min_angle:
            min_angle = angle
            best_o_idx = c_idx

    # 注意：Python 索引从 0 开始，Quantum ESPRESSO 原子的序号从 1 开始
    qe_atom_number = best_o_idx + 1
    print(f"[√] 最终选定跃迁氧原子序号: {qe_atom_number} (夹角: {np.degrees(min_angle):.2f}°)")

    return qe_atom_number, vac_pos.tolist()


if __name__ == "__main__":
    # 测试代码：你可以将这里的路径替换成你的实际路径
    test_file = r".\5Sc5YSZ_Vac_4Zr\espresso.pwo"
    o_num, vac_coords = find_migration_target(test_file, ref_vector=np.array([1.0, 0.0, 0.0]))
    print(f"填入字典的值 -> [{o_num}, {vac_coords}]")
    pass
