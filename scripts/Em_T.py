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
REF_TEMP_C = 25.0      # 假设之前的计算是在室温或0K结构基础上进行的(在此仅作基准)

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


MPI_ROOT = "/root/autodl-tmp/nvhpc/Linux_x86_64/25.3/comm_libs/12.8/hpcx/latest/ompi"
NVHPC_BASE = "/root/autodl-tmp/nvhpc/Linux_x86_64/25.3"

MPIRUN_PATH = f"{MPI_ROOT}/bin/mpirun"
QE_BIN_DIR = "/root/autodl-tmp/q-e-qe-7.5/bin"
PW_PATH = f"{QE_BIN_DIR}/pw.x"
NEB_PATH = f"{QE_BIN_DIR}/neb.x"

os.environ['OPAL_PREFIX'] = MPI_ROOT
os.environ['PATH'] = f"{MPI_ROOT}/bin:{NVHPC_BASE}/compilers/bin:" + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = f"{MPI_ROOT}/lib:{NVHPC_BASE}/compilers/lib:" + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PSEUDO_ABS_PATH = os.path.join(PROJECT_ROOT, "pseudos")

REQUIRED_PSEUDOS = {
    'Zr': 'Zr.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Sc': 'Sc.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Y':  'Y.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'O':  'O.pbe-n-kjpaw_psl.1.0.0.UPF'
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
                'degauss': DEGAUSS_RY,      # 对应温度的展宽
            },
            'electrons': {
                'conv_thr': 1.0e-4, 
                'mixing_beta': 0.2,
                'electron_maxstep': 100,
                'diagonalization' : 'david',
                'mixing_ndim' : 4
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
            if i == 0: header = "FIRST_IMAGE\n"
            elif i == n_images - 1: header = "LAST_IMAGE\n"
            else: header = "INTERMEDIATE_IMAGE\n"
            
            block = header + "ATOMIC_POSITIONS (angstrom)\n"
            for atom in img:
                if np.isnan(atom.position).any():
                    print(f"Error: NaN detected in image {i+1}")
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
  degauss     = 0.01
/

&ELECTRONS
  conv_thr    = 1.0d-4
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

def find_vacancy_position(perfect_atoms, defect_atoms, threshold=0.8):
    # 这里需要注意：如果 defect_atoms 已经膨胀了，perfect_atoms 也需要膨胀才能对比
    # 或者我们只对比相对坐标。为了简单，假设传入之前已经处理好，或者在此处做容差处理。
    # 鉴于只找空位，可以稍微放宽 threshold
    for atom in perfect_atoms:
        if atom.symbol != 'O': continue
        _, dists_squared = get_distances(
            atom.position[None, :], 
            defect_atoms.positions, 
            cell=defect_atoms.cell, 
            pbc=defect_atoms.pbc
        )
        min_dist = np.sqrt(np.min(dists_squared))
        if min_dist > threshold:
            return atom.position
    return None

# ==========================================
# 3. 主逻辑
# ==========================================

def process_one_case(folder_path):
    case_name = os.path.basename(folder_path)
    # 修改输出目录名以区分高温计算
    neb_dir = folder_path.replace("_Vac_", f"_NEB_{int(TARGET_TEMP_C)}C_")
    os.makedirs(neb_dir, exist_ok=True)
    
    is_tmp  = os.path.join(neb_dir, "tmp_is") # 新增 IS 弛豫目录
    fs_tmp  = os.path.join(neb_dir, "tmp_fs")
    neb_tmp = os.path.join(neb_dir, "tmp_neb")
    
    os.makedirs(is_tmp,  exist_ok=True)
    os.makedirs(fs_tmp,  exist_ok=True)
    os.makedirs(neb_tmp, exist_ok=True)
    
    is_prefix  = f"is_{case_name}" # 新增
    fs_prefix  = f"fs_{case_name}"
    neb_prefix = f"neb_{case_name}"

    qm = QEManager(neb_dir)

    # --- 1. 加载缺陷结构 (Original IS) ---
    pwo_path = os.path.join(folder_path, "espresso.pwo")
    if not os.path.exists(pwo_path): return

    print(f"\n>>> 正在处理: {case_name} @ {TARGET_TEMP_C} C")
    try:
        atoms_is_raw = read(pwo_path, format='espresso-out', index=-1)
    except Exception as e:
        print(f"    ❌ 读取 IS 失败: {e}")
        return

    # --- [关键步骤] 2. 应用热膨胀并重新 Relax IS ---
    # 因为晶胞变大了，原有的原子坐标不再是平衡位置，必须重新弛豫 IS
    atoms_is_expanded = apply_thermal_expansion(atoms_is_raw, TARGET_TEMP_C)
    
    is_pwi = qm.generate_relax_input(atoms_is_expanded, is_prefix, is_tmp)
    is_pwo = os.path.join(neb_dir, f"{is_prefix}.out")
    
    # 检查是否已经算过 IS
    if not (os.path.exists(is_pwo) and "JOB DONE" in open(is_pwo, errors='ignore').read()):
        print("    🚀 Relaxing Initial State (IS) with Thermal Expansion...")
        run_cmd(f"mpirun -np 4 {PW_PATH} -input {is_pwi}", is_pwo)
    else:
        print("    ✅ IS 已经弛豫过 (高温)，跳过。")
        
    # 读取重新弛豫后的 IS
    try:
        atoms_is = read(is_pwo, format='espresso-out')
        print(f"    ✅ IS (高温) 读取成功。")
    except:
        print("    ❌ IS Relax 失败，无法进行后续步骤。")
        return

    # --- 3. 自动定位对应的 Perfect 文件夹 (用于找空位) ---
    material_prefix = case_name.split('_')[0] 
    parent_dir = os.path.dirname(folder_path) 
    perfect_case_name = f"{material_prefix}_Perfect"
    perfect_pwo_path = os.path.join(parent_dir, perfect_case_name, "espresso.pwo")

    vac_pos = None
    if os.path.exists(perfect_pwo_path):
        try:
            atoms_perf = read(perfect_pwo_path, format='espresso-out', index=-1)
            # 注意：Perfect 结构是 0K 的，而 atoms_is 是膨胀后的。
            # 为了对比找空位，我们也需要暂时膨胀 Perfect 结构
            atoms_perf = apply_thermal_expansion(atoms_perf, TARGET_TEMP_C)
            vac_pos = find_vacancy_position(atoms_perf, atoms_is)
        except Exception as e:
            print(f"    ⚠️ 读取完美结构失败: {e}")
    else:
        print(f"    ⚠️ 未找到完美结构: {perfect_case_name}")

    if vac_pos is None:
        print("    ❌ 无法定位空位坐标，尝试使用最近邻推断...")
        # 备选方案：如果找不到，可能因为膨胀导致匹配失败
        # 这里可以加入手动指定空位逻辑，暂略
        return

    # --- 4. 寻找跳跃原子 (FS 构建) ---
    o_indices = [a.index for a in atoms_is if a.symbol == 'O']
    o_positions = atoms_is.positions[o_indices]
    _, d2_to_vac = get_distances(vac_pos[None, :], o_positions, cell=atoms_is.cell, pbc=atoms_is.pbc)
    dists_to_vac = np.sqrt(d2_to_vac.flatten())
    
    min_idx_in_o_list = np.argmin(dists_to_vac)
    moving_atom_idx = o_indices[min_idx_in_o_list] 
    moving_dist = dists_to_vac[min_idx_in_o_list]
    
    print(f"    Feature: 选定跳跃原子 ID={moving_atom_idx}, 距离={moving_dist:.3f} A")
    
    if moving_dist > 3.8: # 高温下距离会变大，稍微放宽阈值
        print("    ⚠️ 警告: 氧原子距离过远，跳过。")
        return

    # --- 5. 构建 FS (Final State) ---
    diff = vac_pos - atoms_is.positions[moving_atom_idx]
    cell = atoms_is.get_cell()
    diff_frac = np.dot(diff, np.linalg.inv(cell))
    diff_frac = diff_frac - np.round(diff_frac)
    real_move = np.dot(diff_frac, cell)
    
    atoms_fs = atoms_is.copy()
    atoms_fs.positions[moving_atom_idx] = atoms_is.positions[moving_atom_idx] + real_move

    # 6. Relax FS (高温)
    fs_pwi = qm.generate_relax_input(atoms_fs, fs_prefix, fs_tmp)
    if fs_pwi is None: return

    fs_pwo = os.path.join(neb_dir, f"{fs_prefix}.out")
    
    if not (os.path.exists(fs_pwo) and "JOB DONE" in open(fs_pwo, errors='ignore').read()):
        print("    🚀 Relaxing Final State (FS)...")
        run_cmd(f"mpirun -np 4 {PW_PATH} -input {fs_pwi}", fs_pwo)
    
    try:
        atoms_fs_relaxed = read(fs_pwo, format='espresso-out')
    except:
        print("    ❌ FS Relax 失败")
        return

    # ... (前面的代码保持不变) ...

    # ==========================================
    # 7. 运行 NEB (高温) - [修改点] 增加插值点
    # ==========================================
    
    # 设定总 image 数量
    n_images_total = 5 
    
    # 构建包含中间点的列表
    # 1. 放入起点
    images = [atoms_is] 
    # 2. 放入 (N-2) 个占位符 (可以是起点的副本)
    for _ in range(n_images_total - 2):
        images.append(atoms_is.copy())
    # 3. 放入终点
    images.append(atoms_fs_relaxed)

    # 使用 ASE 的 NEB 工具进行线性插值
    # 这会计算中间图像的坐标，使它们均匀分布在 IS 和 FS 之间
    neb_tool = NEB(images)
    neb_tool.interpolate() 
    # 如果安装了 IDPP 算法库，也可以用 neb_tool.interpolate('idpp') 获得更好的初猜

    # 生成 NEB 输入文件
    # 注意：这里传入的是包含 5 个结构的 images 列表
    neb_pwi = qm.generate_neb_input(
        images, 
        neb_prefix,
        neb_tmp
    )
    
    if neb_pwi is None: return
    
    neb_out = os.path.join(neb_dir, f"{neb_prefix}.out")
    
    if not (os.path.exists(neb_out) and "JOB DONE" in open(neb_out, errors='ignore').read()):
        # 修改打印信息以确认数量
        print(f"    🚀 Running CI-NEB (800 C) with {n_images_total} images...")
        # 注意：CI-NEB 通常并行效率较低，image 较多时可以适当增加 -np 核数
        run_cmd(f"mpirun -np 1 {NEB_PATH} -input {neb_pwi}", neb_out)
    else:
        print("    ✅ NEB 已经计算过，跳过。")

def main():
    if not os.path.exists(PREVIOUS_RUN_DIR):
        print(f"❌ 错误: 目录 {PREVIOUS_RUN_DIR} 不存在！")
        return

    search_path = os.path.join(PREVIOUS_RUN_DIR, f"*{FILTER_STR}*")
    all_folders = [f for f in glob.glob(search_path) if os.path.isdir(f)]

    order_priority = ["9Sc1YSZ",  "7Sc3YSZ", "6Sc4YSZ", "5Sc5YSZ"]
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
