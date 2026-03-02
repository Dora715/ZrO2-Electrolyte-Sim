# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import subprocess
import time
import glob
import inspect
from ase.io import read, write
from ase.mep import NEB
from ase.geometry.analysis import Analysis
from ase.geometry import get_distances

# ==========================================
# 1. 配置区域
# ==========================================
PREVIOUS_RUN_DIR = "./test_data"
FILTER_STR = "Vac"

# --- 温度效应配置 (新增) ---
TARGET_TEMP_C = 800.0  # 目标温度 800摄氏度
# 氧化锆基材料的线性热膨胀系数 (CTE) 约为 10.5 * 10^-6 / K
# 参考文献: Thermal expansion of YSZ, typically 10-11 ppm/K
CTE = 10.5e-6
REF_TEMP_C = 25.0  # 假设之前的计算是在室温或0K结构基础上进行的(在此仅作基准)

# 计算物理温度对应的 degauss (Ry)
# 1 Ry = 13.605698 eV, kB = 8.617e-5 eV/K
KB_EV = 8.617333e-5
RY_TO_EV = 13.605698
TEMP_K = TARGET_TEMP_C + 273.15
KT_EV = KB_EV * TEMP_K
# 对应 800C 的 degauss (Ry)。注意：真实温度的展宽很小(~0.007 Ry)，
# 有时为了收敛会人为调大，但为了物理意义，这里我们可以设为真实值或保留 0.02
# 这里我们采用 Max(真实温度, 0.02) 以保证收敛性，或者直接使用 Fermi-Dirac 真实温度
# 为了模拟真实高温，建议尝试真实温度，但如果主要关注晶格膨胀，保留 0.02 Gaussian 也可以。
# 下面代码将使用 Fermi-Dirac 分布，且 degauss 设为 max(kT, 0.01) 避免过小不收敛
DEGAUSS_RY = max(KT_EV / RY_TO_EV, 0.01)

print(f"--- 温度设置: {TARGET_TEMP_C} C ({TEMP_K} K) ---")
print(f"--- 电子温度 (degauss): {DEGAUSS_RY:.6f} Ry ---")
print(f"--- 线性膨胀系数: {CTE} ---")

MPI_ROOT = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/openmpi4/openmpi-4.1.5"
NVHPC_BASE = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.3"

MPIRUN_PATH = f"{MPI_ROOT}/bin/mpirun"
QE_BIN_DIR = "/root/autodl-tmp/q-e-qe-7.5/bin"
PW_PATH = f"{QE_BIN_DIR}/pw.x"
NEB_PATH = f"{QE_BIN_DIR}/neb.x"

os.environ['OPAL_PREFIX'] = MPI_ROOT
os.environ['PATH'] = f"{MPI_ROOT}/bin:{NVHPC_BASE}/compilers/bin:" + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = f"{MPI_ROOT}/lib:{NVHPC_BASE}/compilers/lib:" + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PSEUDO_ABS_PATH = os.path.join(PROJECT_ROOT, "pseudos")

REQUIRED_PSEUDOS = {
    'Zr': 'Zr.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Sc': 'Sc.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Y': 'Y.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'O': 'O.pbe-n-kjpaw_psl.1.0.0.UPF'
}


# ==========================================
# 2. 辅助函数
# ==========================================

def apply_thermal_expansion(atoms, target_temp_c, ref_temp_c=25.0, cte=10.5e-6):
    """
    应用热膨胀到晶胞。
    L(T) = L0 * (1 + alpha * delta_T)
    """
    delta_T = target_temp_c - ref_temp_c
    if delta_T <= 0:
        return atoms

    scaling_factor = 1.0 + (cte * delta_T)
    print(f"    🔥 应用热膨胀: Delta T = {delta_T} K, 缩放因子 = {scaling_factor:.6f}")

    # 缩放晶胞和原子坐标
    new_cell = atoms.get_cell() * scaling_factor
    atoms.set_cell(new_cell, scale_atoms=True)
    return atoms


class QEManager:
    def __init__(self, run_dir):
        self.pseudo_dir = PSEUDO_ABS_PATH
        if not os.path.exists(self.pseudo_dir):
            os.makedirs(self.pseudo_dir)

    def check_pseudos(self, atoms):
        species = set(atoms.get_chemical_symbols())
        missing = []
        for s in species:
            if s in REQUIRED_PSEUDOS:
                fpath = os.path.join(self.pseudo_dir, REQUIRED_PSEUDOS[s])
                if not os.path.exists(fpath):
                    missing.append(REQUIRED_PSEUDOS[s])

        if missing:
            print(f"    🚨 错误: 伪势目录 {self.pseudo_dir} 下缺少: {missing}")
            return False
        return True

    def generate_relax_input(self, atoms, prefix, outdir):
        """生成 Relax 输入 (用于 IS 和 FS 的高温弛豫)"""
        if not self.check_pseudos(atoms): return None

        species = sorted(list(set(atoms.get_chemical_symbols())))
        needed_pseudos = {k: REQUIRED_PSEUDOS[k] for k in species}

        # 修改: 使用 Fermi-Dirac 和计算出的 degauss
        input_data = {
            'control': {
                'calculation': 'relax',
                'nstep': 150,
                'prefix': prefix,
                'pseudo_dir': self.pseudo_dir,
                'outdir': outdir,
                'disk_io': 'low',
                'tstress': True,
                'tprnfor': True,
            },
            'system': {
                'ecutwfc': 35, 'ecutrho': 280,
                # --- 修改: 适应高温 ---
                'occupations': 'smearing',
                'smearing': 'fermi-dirac',  # 更符合物理意义的分布
                'degauss': DEGAUSS_RY,  # 对应温度的展宽
            },
            'electrons': {
                'conv_thr': 1.0e-4,
                'mixing_beta': 0.2,
                'electron_maxstep': 100,
                'diagonalization': 'david',
                'mixing_ndim': 4
            }
        }

        outfile = os.path.join(outdir, f'{prefix}.pwi')
        write(outfile, atoms, format='espresso-in',
              input_data=input_data, pseudopotentials=needed_pseudos, kpts=(1, 1, 1))
        return outfile

    def generate_neb_input(self, images, prefix, outdir):
        # [修复1] 先定义输出文件名，防止后面 NameError
        outfile = os.path.join(outdir, f"{prefix}.in")

        # 1. 检查赝势文件是否存在 (使用全局配置)
        if not self.check_pseudos(images[0]): return None

        # 2. 定义原子质量
        ATOMIC_MASSES = {'Zr': 91.224, 'Sc': 44.956, 'Y': 88.906, 'O': 15.999}
        species = sorted(set(images[0].get_chemical_symbols()))

        # 3. 格式化 ATOMIC_SPECIES
        # [修改点] 直接使用全局变量 REQUIRED_PSEUDOS，不再重复定义
        atomic_species_lines = []
        for s in species:
            mass = ATOMIC_MASSES.get(s, 1.0)
            # 直接从全局配置获取文件名，如果找不到则默认 s.UPF
            filename = REQUIRED_PSEUDOS.get(s, f"{s}.UPF")

            line = f" {s:<3} {mass:12.6f} {filename}"
            atomic_species_lines.append(line)
        atomic_species_str = "\n".join(atomic_species_lines)

        # 4. 格式化 CELL_PARAMETERS (强制15位宽，防止粘连)
        cell = images[0].get_cell()
        cell_str = "CELL_PARAMETERS (angstrom)\n" + "\n".join(
            f" {v[0]:15.9f} {v[1]:15.9f} {v[2]:15.9f}" for v in cell
        )

        # 5. 格式化 ATOMIC_POSITIONS (强制15位宽，防止粘连)
        pos_blocks = []
        n_images = len(images)
        for i, img in enumerate(images):
            if i == 0:
                header = "FIRST_IMAGE\n"
            elif i == n_images - 1:
                header = "LAST_IMAGE\n"
            else:
                header = "INTERMEDIATE_IMAGE\n"

            block = header + "ATOMIC_POSITIONS (angstrom)\n"
            for atom in img:
                if np.isnan(atom.position).any():
                    print(f"Error: NaN detected in image {i + 1}")
                    return None
                block += f"{atom.symbol:<3} {atom.position[0]:15.9f} {atom.position[1]:15.9f} {atom.position[2]:15.9f}\n"
            pos_blocks.append(block)

        all_pos_str = "\n".join(pos_blocks)

        # 6. 生成内容 (无中文注释)
        content = f"""BEGIN
BEGIN_PATH_INPUT
&PATH
  restart_mode   = 'from_scratch'
  string_method  = 'neb'
  num_of_images  = {len(images)}
  nstep_path     = 100
  ds             = 0.5
  opt_scheme     = 'quick-min'
  CI_scheme      = 'auto'
  path_thr       = 0.2
/
END_PATH_INPUT

BEGIN_ENGINE_INPUT
&CONTROL
  calculation = 'scf'
  prefix      = '{prefix}'
  outdir      = '{outdir}'
  pseudo_dir  = '{self.pseudo_dir}'
  disk_io     = 'low'
/

&SYSTEM
  ibrav = 0,
  nat   = {len(images[0])},
  ntyp  = {len(species)},
  ecutwfc = 35,
  ecutrho = 280,
  occupations = 'smearing',
  smearing    = 'fermi-dirac',
  degauss     = {DEGAUSS_RY:.6f}
/

&ELECTRONS
  conv_thr    = 1.0d-6
  mixing_beta = 0.2
  electron_maxstep = 200
/

{cell_str}

ATOMIC_SPECIES
{atomic_species_str}

K_POINTS gamma

BEGIN_POSITIONS
{all_pos_str}
END_POSITIONS

END_ENGINE_INPUT
END
"""
        with open(outfile, "w") as f:
            f.write(content)
        return outfile


def run_cmd(cmd, outfile):
    print(f"    Running: {cmd.split()[0]} ...")
    with open(outfile, "w") as f:
        subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=os.environ)


def find_vacancy_position(perfect_atoms, defect_atoms, threshold=1.0):
    """
    使用分数坐标对比，精准定位空位。
    threshold: 判定为空位的最小距离阈值 (Angstrom)
    """
    # 获取晶胞矩阵
    cell = perfect_atoms.get_cell()
    # 转换到分数坐标 (0 to 1)
    perf_scaled = perfect_atoms.get_scaled_positions()
    def_scaled = defect_atoms.get_scaled_positions()

    # 过滤出完美结构中的氧原子索引
    o_indices = [i for i, sym in enumerate(perfect_atoms.symbols) if sym == 'O']

    for idx in o_indices:
        p_pos = perf_scaled[idx]

        # 计算该点到 defect_atoms 中所有原子的位移向量
        diff = def_scaled - p_pos
        # 关键：考虑 PBC 带来的“镜像”距离，将差异限制在 [-0.5, 0.5]
        diff = diff - np.round(diff)

        # 将分数位移转回笛卡尔位移，并计算距离
        dist_matrix = np.linalg.norm(np.dot(diff, cell), axis=1)

        # 如果这个完美结构的氧原子点，到缺陷结构所有原子的距离都大于阈值
        # 说明这个位置就是被挖掉的“空位”
        if np.min(dist_matrix) > threshold:
            print(f"    🎯 成功定位空位: 索引 {idx}, 分数坐标 {p_pos}")
            return perfect_atoms.positions[idx]

    return None


# ==========================================
# 3. 主逻辑
# ==========================================

def process_one_case(folder_path):
    case_name = os.path.basename(folder_path)
    
    # 1. 定义主输出目录 (Base Directory)
    neb_base_dir = folder_path.replace("_Vac_", f"_NEB_{int(TARGET_TEMP_C)}C_")
    os.makedirs(neb_base_dir, exist_ok=True)
    
    # 初始化 QE 管理器 (Base dir 仅用于检查伪势，实际运行会指定子目录)
    qm = QEManager(neb_base_dir)

    # =========================================================================
    # --- 1. 准备初态 (IS) : 读取 -> 热膨胀 -> 弛豫 ---
    # =========================================================================
    pwo_path = os.path.join(folder_path, "espresso.pwo")
    if not os.path.exists(pwo_path): return

    print(f"\n{'='*60}")
    print(f"🏗️  正在处理结构: {case_name} @ {TARGET_TEMP_C} C")
    
    # 读取原始 PWO
    try:
        atoms_is_raw = read(pwo_path, format='espresso-out', index=-1)
    except Exception as e:
        print(f"    ❌ 读取 IS 失败: {e}")
        return

    # 应用热膨胀
    atoms_is_expanded = apply_thermal_expansion(atoms_is_raw, TARGET_TEMP_C)

    # 准备 IS 弛豫目录
    is_dir = os.path.join(neb_base_dir, "IS_Relax")
    os.makedirs(is_dir, exist_ok=True)
    is_prefix = f"is_{case_name}"
    is_pwo = os.path.join(is_dir, f"{is_prefix}.out")

    # 检查或执行 IS 弛豫
    if not (os.path.exists(is_pwo) and "JOB DONE" in open(is_pwo, errors='ignore').read()):
        print("    🚀 Relaxing Initial State (IS) with Thermal Expansion...")
        is_pwi = qm.generate_relax_input(atoms_is_expanded, is_prefix, is_dir)
        run_cmd(f"{MPIRUN_PATH} -np 4 {PW_PATH} -input {is_pwi}", is_pwo)
    else:
        print("    ✅ IS 已经弛豫过 (高温)，跳过。")

    # 读取弛豫后的 IS
    try:
        atoms_is = read(is_pwo, format='espresso-out')
    except:
        print("    ❌ IS Relax 结果读取失败，无法继续。")
        return

    # =========================================================================
    # --- 2. 定位空位 (Vacancy Location) ---
    # =========================================================================
    material_prefix = case_name.split('_')[0]
    parent_dir = os.path.dirname(folder_path)
    perfect_case_name = f"{material_prefix}_Perfect"
    perfect_pwo_path = os.path.join(parent_dir, perfect_case_name, "espresso.pwo")

    vac_pos = None
    if os.path.exists(perfect_pwo_path):
        try:
            atoms_perf = read(perfect_pwo_path, format='espresso-out', index=-1)
            atoms_perf = apply_thermal_expansion(atoms_perf, TARGET_TEMP_C)
            vac_pos = find_vacancy_position(atoms_perf, atoms_is)
        except Exception as e:
            print(f"    ⚠️ 读取完美结构失败: {e}")
    
    if vac_pos is None:
        print("    ❌ 无法定位空位坐标，跳过此任务。")
        return

    # =========================================================================
    # --- 3. [智能分析] 环境识别与路径扫描 (Environment & Gate Analysis) ---
    # =========================================================================
    
    # A. 识别空位环境 (Initial State Environment)
    # -----------------------------------------------------------
    cation_indices = [a.index for a in atoms_is if a.symbol in ['Zr', 'Sc', 'Y']]
    cation_pos = atoms_is.positions[cation_indices]
    
    # 计算空位到阳离子的距离，找最近的4个
    _, vac_cat_dists = get_distances(vac_pos[None, :], cation_pos, cell=atoms_is.cell, pbc=True)
    vac_cat_dists = vac_cat_dists.flatten()
    sorted_cat_idx = np.argsort(vac_cat_dists)
    nearest_4_indices = [cation_indices[i] for i in sorted_cat_idx[:4]]
    nearest_4_syms = [atoms_is[i].symbol for i in nearest_4_indices]
    
    # 生成环境标签 (如 "1Sc3Zr")
    env_counts = collections.Counter(nearest_4_syms)
    env_parts = [f"{count}{sym}" for sym, count in sorted(env_counts.items())] 
    vac_env_label = "".join(env_parts)
    print(f"    📍 空位环境 (Initial Env): 【 {vac_env_label} 】")

    # B. 扫描所有潜在跳跃路径
    # -----------------------------------------------------------
    o_indices = [a.index for a in atoms_is if a.symbol == 'O']
    potential_tasks = []

    print(f"    🔍 正在扫描并分类迁移路径...")
    
    for o_idx in o_indices:
        o_pos = atoms_is.positions[o_idx]
        # 计算跳跃距离
        dist = get_distances(o_pos[None, :], vac_pos[None, :], cell=atoms_is.cell, pbc=True)[1][0][0]
        
        # 筛选合理的跳跃距离 (1.8 - 3.5 Å)
        if 1.8 < dist < 3.5:
            # --- 识别门 (Gate Identification) ---
            # 计算中点 (鞍点)
            midpoint = (o_pos + vac_pos) / 2.0
            
            # 计算中点到所有阳离子距离，找最近的2个
            _, gate_dists = get_distances(midpoint[None, :], cation_pos, cell=atoms_is.cell, pbc=True)
            gate_dists = gate_dists.flatten()
            sorted_gate_idx = np.argsort(gate_dists)
            
            gate_sym1 = atoms_is[cation_indices[sorted_gate_idx[0]]].symbol
            gate_sym2 = atoms_is[cation_indices[sorted_gate_idx[1]]].symbol
            
            # 排序生成门标签 (如 "Sc-Zr")
            gate_label = "-".join(sorted([gate_sym1, gate_sym2]))
            
            # 记录任务
            potential_tasks.append({
                'o_index': o_idx,
                'dist': dist,
                'gate': gate_label,
                'env': vac_env_label
            })

    # C. 去重逻辑 (Filter Unique Paths)
    # -----------------------------------------------------------
    final_tasks = {}
    for task in potential_tasks:
        gate = task['gate']
        # 如果这种门还没收录，或者新的这个距离更接近 2.55 (理想值)，则更新
        if gate not in final_tasks:
            final_tasks[gate] = task
        else:
            current_best = final_tasks[gate]
            if abs(task['dist'] - 2.55) < abs(current_best['dist'] - 2.55):
                final_tasks[gate] = task

    print(f"    ✅ 筛选完成! 发现 {len(potential_tasks)} 个候选，去重后将计算 {len(final_tasks)} 条不同路径。")

    # =========================================================================
    # --- 4. [循环执行] 遍历所有唯一路径并计算 (Loop over Tasks) ---
    # =========================================================================
    
    for gate_type, task in final_tasks.items():
        moving_idx = task['o_index']
        dist_val = task['dist']
        env_str = task['env']
        
        print(f"\n    👉 [任务处理] Env: {env_str} | Gate: {gate_type} | Atom ID: {moving_idx} (d={dist_val:.2f}A)")
        
        # --- 4.1 创建专属子目录 ---
        # 结构: NEB_800C/10ScSZ_Vac_1/1Sc3Zr_Env/Zr-Zr_Gate/
        path_dir_name = os.path.join(env_str + "_Env", gate_type + "_Gate")
        task_dir = os.path.join(neb_base_dir, path_dir_name)
        os.makedirs(task_dir, exist_ok=True)
        
        # --- 4.2 构建并弛豫 FS (Final State) ---
        fs_prefix = f"fs_{gate_type}" # 前缀加上门类型，便于区分
        fs_pwo = os.path.join(task_dir, f"{fs_prefix}.out")
        
        # 构建 FS 结构
        atoms_fs = atoms_is.copy()
        # 计算并应用位移
        vec_to_vac = get_distances(atoms_is.positions[moving_idx][None, :], vac_pos[None, :], 
                                   cell=atoms_is.cell, pbc=True)[0][0]
        atoms_fs.positions[moving_idx] += vec_to_vac
        
        # 检查是否已弛豫 FS
        fs_relaxed = None
        if not (os.path.exists(fs_pwo) and "JOB DONE" in open(fs_pwo, errors='ignore').read()):
            print(f"       🔨 Relaxing Final State for {gate_type} path...")
            fs_pwi = qm.generate_relax_input(atoms_fs, fs_prefix, task_dir)
            run_cmd(f"{MPIRUN_PATH} -np 4 {PW_PATH} -input {fs_pwi}", fs_pwo)
        else:
            print(f"       ✅ FS ({gate_type}) 已经弛豫过。")
            
        # 读取弛豫后的 FS
        try:
            fs_relaxed = read(fs_pwo, format='espresso-out')
        except:
            print(f"       ❌ 读取 FS ({gate_type}) 失败，跳过此路径。")
            continue # 跳过当前循环，去算下一个门
            
        # --- 4.3 准备并运行 NEB ---
        neb_prefix = f"neb_{gate_type}"
        neb_out = os.path.join(task_dir, f"{neb_prefix}.out")
        
        if os.path.exists(neb_out) and "JOB DONE" in open(neb_out, errors='ignore').read():
            print(f"       ✅ NEB ({gate_type}) 已经计算完毕。")
            continue

        print(f"       🚀 Preparing NEB for {gate_type} path...")
        
        # 插值生成图像
        n_images_total = 5 # 或者 7
        images = [atoms_is]
        for _ in range(n_images_total - 2):
            images.append(atoms_is.copy())
        images.append(fs_relaxed)
        
        # NEB 插值
        neb_tool = NEB(images)
        try:
            neb_tool.interpolate('idpp')
        except:
            neb_tool.interpolate() # 回退到线性插值
            
        # 生成输入并运行
        neb_pwi = qm.generate_neb_input(images, neb_prefix, task_dir)
        if neb_pwi:
            run_cmd(f"{MPIRUN_PATH} -np 4 {NEB_PATH} -input {neb_pwi}", neb_out) # 注意 NEB 核数调整
        
    print(f"    🏁 完成结构 {case_name} 的所有路径计算。")


def main():
    if not os.path.exists(PREVIOUS_RUN_DIR):
        print(f"❌ 错误: 目录 {PREVIOUS_RUN_DIR} 不存在！")
        return

    search_path = os.path.join(PREVIOUS_RUN_DIR, f"*{FILTER_STR}*")
    all_folders = [f for f in glob.glob(search_path) if os.path.isdir(f)]

    order_priority = ["10ScSZ","9Sc1YSZ","8Sc2YSZ", "7Sc3YSZ", "6Sc4YSZ", "5Sc5YSZ"]

    def sort_key(folder_path):
        name = os.path.basename(folder_path)
        for i, key in enumerate(order_priority):
            if key in name: return i
        return 999

    sorted_folders = sorted(all_folders, key=sort_key)
    print(f"找到 {len(sorted_folders)} 个任务文件夹 (Target Temp: {TARGET_TEMP_C} C)")

    for folder in sorted_folders:
        process_one_case(folder)

    print("\n所有高温 NEB 任务处理完毕。")


if __name__ == "__main__":
    main()
