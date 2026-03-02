import os
import sys
import numpy as np
import random
import subprocess  # 导入子进程模块，用于在 Python 中执行外部命令 (如 mpirun, pw.x)
import time
import multiprocessing as mp
from datetime import datetime
from collections import Counter
from ase import Atoms

# ==========================================
# 1. 环境配置 (针对 A100 本地编译版优化)
# ==========================================
MPI_ROOT = "/root/autodl-tmp/nvhpc/Linux_x86_64/25.3/comm_libs/12.8/hpcx/latest/ompi"
NVHPC_BASE = "/root/autodl-tmp/nvhpc/Linux_x86_64/25.3"

os.environ['OPAL_PREFIX'] = MPI_ROOT
os.environ['PATH'] = f"{MPI_ROOT}/bin:{NVHPC_BASE}/compilers/bin:" + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = f"{MPI_ROOT}/lib:{NVHPC_BASE}/compilers/lib:" + os.environ.get('LD_LIBRARY_PATH', '')

os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
# 3. 修正 QE 路径（确保指向 bin/pw.x）
QE_PATH = "/root/autodl-tmp/q-e-qe-7.5/bin/pw.x"

# --- B. 硬件资源控制 ---
omp_threads = '2'

# 注入环境变量
os.environ['OMP_NUM_THREADS'] = omp_threads
os.environ['MKL_NUM_THREADS'] = omp_threads

# --- C. 日志与缓冲优化 (保持不变) ---
os.environ['PYTHONUNBUFFERED'] = '1'          # Python 实时输出
os.environ['GFORTRAN_UNBUFFERED_ALL'] = 'y'   # Fortran 实时输出
os.environ['OMPI_MCA_orte_base_help_aggregate'] = '0' # 减少 MPI 报错刷屏

# --- D.  任务并发控制 ---
NUM_PARALLEL_TASKS = 2

# ==========================================
# 2. 核心类定义
# ==========================================

class ZrO2Builder:
    """构建氧化锆晶体结构的类"""

    def __init__(self, supercell_size=(2, 2, 2)):
        from ase.build import bulk  # 延迟导入 ASE 的 bulk 函数，用于创建体材料
        self.supercell_matrix = np.diag(supercell_size)  # 创建扩胞矩阵 (对角矩阵)，例如 2x2x2
        self.base = bulk('ZrO2', 'fluorite', a=5.125, cubic=True)  # 创建 ZrO2 的萤石结构基元，晶格常数设为 5.125 Å

    def build_doped_structure(self, name, n_Zr, n_Sc, n_Y):
        from ase.build import make_supercell  # 导入扩胞函数
        atoms = make_supercell(self.base, self.supercell_matrix)  # 基于基元和扩胞矩阵生成超胞
        cation_indices = [a.index for a in atoms if a.symbol == 'Zr']  # 获取所有阳离子 (Zr) 的索引列表
        total_cations_req = n_Zr + n_Sc + n_Y  # 计算配方中要求的总阳离子比例份数
        total_cations_actual = len(cation_indices)  # 获取超胞中实际存在的总阳离子数量

        # 根据比例计算需要掺杂的 Sc 目标原子数 (四舍五入)
        n_sc_target = int(round(total_cations_actual * (n_Sc / total_cations_req)))
        # 根据比例计算需要掺杂的 Y 目标原子数 (四舍五入)
        n_y_target = int(round(total_cations_actual * (n_Y / total_cations_req)))

        # 打印构建信息：显示目标掺杂数量和实际总阳离子数
        print(f"[{name}] 模型构建: Sc={n_sc_target}, Y={n_y_target} (Total Cations={total_cations_actual})")

        # 随机选择 Sc 的掺杂位置
        sc_indices = random.sample(cation_indices, n_sc_target)
        for idx in sc_indices: atoms[idx].symbol = 'Sc'  # 将选中的 Zr 原子替换为 Sc

        # 找出剩余未被 Sc 替换的 Zr 原子索引
        remaining_zr = list(set(cation_indices) - set(sc_indices))
        # 确保 Y 的掺杂数量不超过剩余 Zr 的数量 (防止越界)
        if n_y_target > len(remaining_zr): n_y_target = len(remaining_zr)

        # 随机选择 Y 的掺杂位置
        y_indices = random.sample(remaining_zr, n_y_target)
        for idx in y_indices: atoms[idx].symbol = 'Y'  # 将选中的 Zr 原子替换为 Y

        return atoms  # 返回构建好的 ASE Atoms 对象


class QEManager:
    """管理 Quantum ESPRESSO 输入输出"""

    def __init__(self, project_root, pseudo_dir):
        self.root = project_root  # 设置项目根目录
        self.pseudo_dir = pseudo_dir  # 获取伪势文件夹的绝对路径
        # 定义元素与伪势文件名的映射字典
        self.pseudopotentials = {
            'Zr': 'Zr.pbe-spn-kjpaw_psl.1.0.0.UPF',
            'Sc': 'Sc.pbe-spn-kjpaw_psl.1.0.0.UPF',
            'Y': 'Y.pbe-spn-kjpaw_psl.1.0.0.UPF',
            'O': 'O.pbe-n-kjpaw_psl.1.0.0.UPF'
        }

    def generate_input(self, atoms, task_name, calc_dir, override_data=None):
        from ase.io import write  # 导入 ASE 的写入函数

        unique_outdir = os.path.join(calc_dir, f"tmp_{task_name}")
        if not os.path.exists(unique_outdir):
            os.makedirs(unique_outdir)

        docker_pseudo_dir = "/workspace/pseudos"

        # 定义默认的 Quantum ESPRESSO 输入参数字典
        input_data = {
            'control': {
                'calculation': 'relax',
                'nstep': 100,
                'etot_conv_thr': 1.0e-4,
                'forc_conv_thr': 1.0e-3,
                'restart_mode': 'from_scratch',
                'prefix': f'calc_{task_name}',
                'pseudo_dir': './pseudos',
                'outdir': './tmp', 
                'tprnfor': True,
                'disk_io': 'none',
                'verbosity': 'high'
            },
            'system': {
                'ecutwfc': 60,
                'ecutrho': 480,  
                'occupations': 'smearing',
                'smearing': 'gaussian',
                'degauss': 0.005,
            },
            'electrons': {
                'conv_thr': 1.0e-6,
                'mixing_beta': 0.3,  
                'electron_maxstep': 100,
                'diagonalization': 'david'
            }
        }

        # 如果传入了 override_data (覆盖数据)，则更新默认参数
        # 这用于特殊计算，例如 O2 分子需要开启自旋 (nspin=2)
        if override_data:
            for section, params in override_data.items():  # 遍历覆盖数据的每个部分 (如 system)
                if section in input_data:
                    input_data[section].update(params)  # 如果该部分已存在，则更新对应的参数
                else:
                    input_data[section] = params  # 如果不存在，则添加整个新部分

        input_file = os.path.join(calc_dir, 'espresso.pwi')  # 定义输入文件的完整路径
        # 根据当前 atoms 对象中包含的元素，筛选出需要的伪势文件
        needed_pseudos = {k: v for k, v in self.pseudopotentials.items() if k in atoms.get_chemical_symbols()}

        # 优化 K 点策略：
        # 如果是孤立体系（原子数少且盒子大），用 Gamma 点
        # 如果是固体超胞，至少使用 2x2x2 K 点以确保能量收敛
        kpts = (1, 1, 1) if len(atoms) <= 2 else (2, 2, 2)

        # 使用 ASE 将结构和参数写入输入文件
        write(input_file, atoms, format='espresso-in',
              input_data=input_data,
              pseudopotentials=needed_pseudos,
              kpts=kpts)  # 设置 K 点网格为 Gamma 点 (1x1x1)
        return input_file  # 返回生成的输入文件路径

# 定义环境指纹识别类 (集成自动晶格修复)
class EnvironmentFingerprinter:
    def __init__(self, ase_atoms):
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.core import Lattice, Structure
        from pymatgen.analysis.local_env import CrystalNN
        import numpy as np

        # 1. 转换结构
        self.structure = AseAtomsAdaptor.get_structure(ase_atoms)

        # 2. [关键修复] 检查晶格是否丢失 (Volume ~ 0)
        if self.structure.volume < 0.1:
            print("   >>> [自动修复] 检测到 Volume=0，正在重构晶格以适配 CrystalNN...")
            coords = self.structure.cart_coords
            # 动态计算盒子大小：最大坐标 + 3.0埃缓冲
            # ScSZ 经验值保底 10.26，防止单原子或小团簇过小
            box_len = max(np.max(coords) + 3.0, 10.26)
            new_lattice = Lattice.from_parameters(box_len, box_len, box_len, 90, 90, 90)
            self.structure = Structure(new_lattice, self.structure.species, coords)

        # 3. 猜测氧化态 (CrystalNN 必需)
        try:
            self.structure.add_oxidation_state_by_guess()
        except:
            # 回退策略：手动指定 ScSZ 典型价态
            self.structure.add_oxidation_state_by_element({"Zr": 4, "Sc": 3, "Y": 3, "O": -2})

        # 4. 初始化分析器
        self.cnn = CrystalNN(weighted_cn=False, cation_anion=True)

    def analyze(self):
        from collections import Counter
        env_groups = {}

        # 合法配位数范围 (萤石结构 O 通常配位数为 4)
        # 允许 3-5 以容纳畸变，过滤掉 0, 1, 2 这种边界截断导致的错误
        VALID_CNS = {3, 4, 5, 6}

        print(f"   >>> [过滤策略] 仅保留配位数为 {VALID_CNS} 的合理环境...")

        for i, site in enumerate(self.structure):
            # 只分析氧原子
            if "O" not in site.specie.symbol: continue

            try:
                # 获取配位环境
                nn = self.cnn.get_nn_info(self.structure, i)
                if not nn: continue

                # 检查总配位数是否合理
                cn = len(nn)
                if cn not in VALID_CNS:
                    # 默默跳过不合理的原子 (Isolated, 1Zr, 2Zr 等)
                    continue

                # 提取邻居元素符号 (去除价态数字和数字后缀)
                # 例如: Zr4+ -> Zr
                syms = ["".join([c for c in n['site'].specie.symbol if c.isalpha()]) for n in nn]
                syms.sort()

                # 生成标签 (如 1Sc_3Zr)
                counts = Counter(syms)
                # 排序逻辑：按元素字母顺序 (Sc, Zr) 拼接，保证唯一性
                # 或者按数量排序: key=lambda x:x[1], reverse=True
                # 这里推荐按元素名排序，标签更稳定
                parts = [f"{v}{k}" for k, v in sorted(counts.items())]
                label = "_".join(parts)

                if label not in env_groups: env_groups[label] = []
                env_groups[label].append(i)

            except Exception as e:
                pass

        return env_groups
# ==========================================
# 3. 实时监控与解析函数
# ==========================================
def run_and_monitor(cmd, output_file_path, task_tag="System"):
    prefix = f"[\033[96m{task_tag}\033[0m]"
    """执行命令并实时打印进度"""
    print(f"{prefix} 🚀 启动任务: {cmd}")
    print(f"{prefix} 📄 日志路径: {output_file_path}")
    print("    ------------------------------------------------------")
    print("    [进度监控] 正在启动计算核心...")

    start_time = time.time()  # 总开始时间
    step_start_time = start_time  # 初始化步骤开始时间（关键修复）
    total_start_time = start_time  # 初始化总耗时计时器（关键修复）
    step_count = 0  # 初始化离子步计数器

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    with open(output_file_path, "w",buffering=1) as f_log:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None: break
            if line:
                f_log.write(line)

                # --- 增强版实时反馈逻辑 ---
                if "Forces acting on atoms" in line:
                    step_count += 1
                    current_time = time.time()
                    duration = current_time - step_start_time
                    print(f"{prefix} >>> [Step {step_count}] 优化中... (本步耗时: {duration:.1f}s)")
                    step_start_time = current_time  # 重置下一步计时器

                if "total energy" in line and "!" in line:
                    try:
                        energy = line.split('=')[1].strip().split()[0]
                        print(f"{prefix}     当前能量: {energy} Ry")
                    except:
                        pass

                if "Total force" in line:
                    try:
                        parts = line.split()
                        force = float(parts[2])
                        status = "🔴" if force > 0.1 else "🟡" if force > 0.05 else "🟢"
                        print(f"{prefix}     当前受力: {force:.6f} {status}")
                    except:
                        pass

                if "JOB DONE" in line:
                    total_duration = time.time() - total_start_time
                    print(f"{prefix} ✅ 计算成功结束! 总耗时: {total_duration / 60:.2f} 分钟")

    rc = process.poll()
    if rc != 0:
        print(f"{prefix} ❌ 任务报错! 请检查日志。")
        raise subprocess.CalledProcessError(rc, cmd)


def parse_energy(filepath):
    """从输出文件中提取最终能量 (eV)"""
    if not os.path.exists(filepath): return None  # 如果文件不存在，返回 None
    enc = None
    with open(filepath, 'r') as f:  # 打开文件
        # 倒序读取所有行 (因为最终能量通常在文件末尾)
        for line in reversed(f.readlines()):
            if "!    total energy" in line:  # 找到包含最终能量的行
                try:
                    # 提取数值并转换单位：Ry (Rydberg) -> eV (Electronvolt)
                    # 1 Ry ≈ 13.6057 eV
                    enc = float(line.split()[-2]) * 13.6057
                    break  # 找到后立即退出循环
                except:
                    pass
    return enc  # 返回提取到的能量 (eV)


def run_single_material_task(task_args):
    """
    单个材料体系的完整流水线函数 (多进程封装版)
    task_args: 包含 (材料名, 掺杂比例, qe_manager, base_dir, chem_pot_o) 的元组
    """
    name, ratios, qe_manager, base_dir, chem_pot_o,gpu_id = task_args
    local_results = []  # 局部结果列表，用于进程间数据收集
    builder = ZrO2Builder()

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    print(f"\n[进程 {os.getpid()} | 显卡 GPU {gpu_id}] >>> 开始处理材料: {name}")

    # --- A. 完美晶胞计算 ---
    # 这里的 struct 是原始的、未产生空位的完美结构
    struct = builder.build_doped_structure(name, ratios['Zr'], ratios['Sc'], ratios['Y'])
    struct.rattle(stdev=0.01, seed=42)

    print(f"\n   >>> [步骤 A: 基准参考] 计算完美晶胞能量...")
    task_perf = f"{name}_Perfect"
    dir_perf = os.path.join(base_dir, task_perf)

    # 生成输入并运行
    inp_perf = qe_manager.generate_input(struct, task_perf, dir_perf)
    out_perf = os.path.join(dir_perf, 'espresso.pwo')

    # GPU 环境下 MAX_MPI_CORES 建议为 1 (由 multiprocessing 控制并行任务数)
    cmd_perf = f"mpirun --allow-run-as-root -np 4 {QE_PATH} -nk 4 -input {inp_perf}"

    try:
        if not (os.path.exists(out_perf) and "JOB DONE" in open(out_perf, errors='ignore').read()):
            print(f"   >>> 计算完美晶胞 (GPU {gpu_id})...")
            run_and_monitor(cmd_perf, out_perf, task_tag=f"{name}_Perf")
        e_perfect = parse_energy(out_perf)
    except Exception as e:
        print(f"    ❌ 完美晶胞执行出错: {e}")
        e_perfect = None

    # --- 逻辑修正点：只有 e_perfect 成功获取才进行后续步骤 ---
    if e_perfect:
        print(f"    ★ E_perfect: {e_perfect:.6f} eV")

        # --- B. 环境分析 (Environment Analysis) ---
        print("\n   >>> [步骤 B: 环境分析] 正在分析氧配位环境 (化学指纹法)...")
        envs = {}
        try:
            fingerprinter = EnvironmentFingerprinter(struct)
            envs = fingerprinter.analyze()
        except Exception as e:
            print(f"   >>> [错误] 环境分析模块崩溃: {e}")

        # 应急处理：防止环境识别为空导致循环无法进行
        if not envs:
            print("   >>> [警告] 未识别到合理环境，强制使用 Index 0 测试。")
            envs = {"Fallback_Random": [0]}

        # 排序并准备计算
        sorted_envs = sorted(envs.items(), key=lambda item: len(item[1]), reverse=True)
        print(f"\n   [环境统计表] 共识别出 {len(envs)} 种环境:")
        for i, (env_name, indices) in enumerate(sorted_envs):
            print(f"   {i + 1:<4} {env_name:<20} {len(indices):<6} {indices[0]}")

        # ==========================================
        # --- C. 缺陷计算 (Defect Cell) ---
        # ==========================================
        for i, (label, indices) in enumerate(sorted_envs):
            idx = indices[0]  # 取该环境下第一个原子

            print(f"\n   -------------------------------------------------------")
            print(f"   >>> [步骤 C: 缺陷计算] ({i + 1}/{len(sorted_envs)}) 类型: {label} (Atom ID: {idx})")

            # 1. 构建缺陷结构 (基于完美晶胞删除一个氧原子)
            defect_struct = struct.copy()
            del defect_struct[idx]

            # 2. 定义任务路径
            task_vac = f"{name}_Vac_{label}"
            dir_vac = os.path.join(base_dir, task_vac)

            # 3. 生成输入文件 (QEManager 内部需处理 unique_outdir)
            inp_vac = qe_manager.generate_input(defect_struct, task_vac, dir_vac)
            out_vac = os.path.join(dir_vac, 'espresso.pwo')

            # 4. 运行计算 (包含 GPU 关键参数 -nb 8)
            cmd_vac = f"mpirun --allow-run-as-root -np 4 {QE_PATH} -nk 4 -input {inp_vac}"

            e_defect = None
            e_final_form = "N/A"

            try:
                if os.path.exists(out_vac) and "JOB DONE" in open(out_vac, errors='ignore').read():
                    print(f"      -> 检测到计算已完成，跳过执行...")
                else:
                    run_and_monitor(cmd_vac, out_vac, task_tag=f"{name}_{label}")

                # 5. 解析能量并计算形成能
                e_defect = parse_energy(out_vac)

                if e_defect is not None and chem_pot_o is not None:
                    e_final_form = e_defect - e_perfect + chem_pot_o
                    print(f"        [最终结果] 类型 {label} | Ef: {e_final_form:.6f} eV")
                else:
                    print("        ⚠️ 数据缺失，无法计算 Ef")

                # 6. 【关键修改】将结果存入本进程的局部列表
                local_results.append({
                    "Material": name,
                    "Environment": label,
                    "Atom_Index": idx,
                    "E_Perfect": e_perfect,
                    "E_Defect": e_defect,
                    "Chemical_Pot_O": chem_pot_o,
                    "Formation_Energy_Ef": e_final_form
                })

            except Exception as e:
                print(f"   ❌ 缺陷计算 {label} 处理出错: {e}")

        print(f"\n[进程 {os.getpid()}] ✅ {name} 所有环境处理完毕。")
    else:
        print(f"    ❌ {name} 完美晶胞计算失败，无法继续缺陷计算。")

    return local_results

# ==========================================
# 4. 主流程
# ==========================================
def main():
    print("===========================================")
    print("   ZrO2 氧空位形成能计算 (全自动版)")
    print("   包含: O2分子 -> 完美晶胞 -> 环境分析 -> 缺陷晶胞 -> Ef")
    print("===========================================")

    
    # 1. 初始化
    # 创建基于当前时间的运行目录名称
    base_dir = f"./FullRun_{datetime.now().strftime('%Y%m%d_%H%M')}"
    pseudo_dir = "./pseudos"  # 定义伪势存放目录
    if not os.path.exists(base_dir): os.makedirs(base_dir)  # 创建运行目录
    if not os.path.exists(pseudo_dir): os.makedirs(pseudo_dir)  # 创建伪势目录

    # 2. 伪势准备
    # 定义需要下载的伪势文件名列表
    files = ['Zr.pbe-spn-kjpaw_psl.1.0.0.UPF', 'Sc.pbe-spn-kjpaw_psl.1.0.0.UPF',
             'Y.pbe-spn-kjpaw_psl.1.0.0.UPF', 'O.pbe-n-kjpaw_psl.1.0.0.UPF']
    base_url = "https://pseudopotentials.quantum-espresso.org/upf_files/"  # 伪势下载基地址
    print(">>> [系统] 检查伪势...")
    for f in files:  # 遍历所需伪势
        path = os.path.join(pseudo_dir, f)  # 伪势完整路径
        if not os.path.exists(path):  # 如果伪势不存在
            # 使用 wget 命令下载伪势，-q 为静默模式，-O 指定输出路径
            subprocess.run(f"wget -q -O {path} {base_url}{f}", shell=True)

    qe_manager = QEManager(base_dir, pseudo_dir)  # 实例化 QE 管理器
    results = []  # 初始化结果列表

    # =======================================================
    # 【步骤 0】 计算氧结合能 Eb = E(O2) - 2*E(O_atom)
    # =======================================================
    print(f"\n########################################")
    print(f"   步骤 0: 计算氧结合能 (终极修正版)")
    print(f"########################################")

    # --- 0.1 计算 O2 分子 ---
    task_o2 = "O2_Ref"
    dir_o2 = os.path.join(base_dir, task_o2)
    # O2 分子稍微偏离中心，防止高对称性干扰
    atoms_o2 = Atoms('O2', positions=[(7.5, 7.5, 7.5), (7.5, 7.5, 8.73)], cell=[15, 15, 15], pbc=True)
    
    o2_settings = {
        'system': {
            'nspin': 2,
            'tot_magnetization': 2.0,  #O2 基态是三重态，净自旋为 2
            'occupations': 'smearing', 
            'smearing': 'gauss',
            'degauss': 0.005,
        },
        'electrons': {
            'mixing_beta': 0.3  # 分子计算标准混合因子
        }
    }

    # --- 0.2 计算孤立 O 原子 ---
    task_atom = "O_Atom_Ref"
    dir_atom = os.path.join(base_dir, task_atom)
    
    # 打破对称性！
    # 不要放在 (6,6,6)，放在歪一点的地方，让 p 轨道分裂
    atoms_atom = Atoms('O', positions=[(6.12, 6.23, 6.34)], cell=[12, 12, 12], pbc=True)
    
    atom_settings = {
        'system': {
            'nspin': 2,
            'tot_magnetization': 2.0,  # 强制总磁矩为 2 (Hund规则)
            'occupations': 'fixed',    # 单原子改用 fixed (如果有报错提示能级交叉，则改回 smearing)
        },
        'electrons': {
            'mixing_beta': 0.1,        # 降低混合因子，防止电荷震荡
            'electron_maxstep': 200
        }
    }

    chem_pot_o=None
    try:
        # 1. 计算 O2
        inp_o2 = qe_manager.generate_input(atoms_o2, task_o2, dir_o2, override_data=o2_settings)
        run_and_monitor(f"mpirun --allow-run-as-root {QE_PATH} -np 1 -nk 1 -input {inp_o2}",
                        os.path.join(dir_o2, 'espresso.pwo'), task_tag="O2_Ref")
        e_o2 = parse_energy(os.path.join(dir_o2, 'espresso.pwo')) # 这里的返回值单位其实是 eV

        # 2. 计算 O 原子
        inp_atom = qe_manager.generate_input(atoms_atom, task_atom, dir_atom, override_data=atom_settings)
        run_and_monitor(f"mpirun --allow-run-as-root {QE_PATH} -np 1 -nk 1 -input {inp_atom}",
                        os.path.join(dir_atom, 'espresso.pwo'), task_tag="O_Atom")
        e_atom = parse_energy(os.path.join(dir_atom, 'espresso.pwo')) # eV

        if e_o2 and e_atom:
            eb = e_o2 - (2 * e_atom)
            chem_pot_o = e_o2 / 2.0
            
            print(f"\n    ----------------------------------------")
            print(f"    E(O2)   = {e_o2:.4f} eV") 
            print(f"    E(Atom) = {e_atom:.4f} eV")
            print(f"    ----------------------------------------")
            print(f"    ★ 氧结合能 Eb: {eb:.4f} eV ") 
            print(f"    ★ 氧化学势 μ_O: {chem_pot_o:.4f} eV")
            print(f"    ----------------------------------------\n")
            
    except Exception as e:
        print(f"    ❌ 氧参考态计算失败: {e}")
        chem_pot_o=None

    # 准备任务列表
    all_materials = [
        ("10ScSZ", {"Zr": 90, "Sc": 20, "Y": 0},"0"),
        ("9Sc1YSZ", {"Zr": 90, "Sc": 18, "Y": 2},"0"),
        ("8Sc2YSZ", {"Zr": 90, "Sc": 16, "Y": 4},"0"),
        ("7Sc3YSZ", {"Zr": 90, "Sc": 14, "Y": 6},"0"),
        ("6Sc4YSZ", {"Zr": 90, "Sc": 12, "Y": 8},"0"),
        ("5Sc5YSZ", {"Zr": 90, "Sc": 10, "Y": 10},"0")
    ]

    # 构造传递给任务函数的参数包
    tasks = []
    for name, ratios, gpu_id in all_materials:
        tasks.append((name, ratios, qe_manager, base_dir, chem_pot_o,gpu_id))

    print(f"\n>>> [并行启动] 使用 Pool 同时启动 {NUM_PARALLEL_TASKS} 个材料计算任务...")

    with mp.Pool(processes=NUM_PARALLEL_TASKS) as pool:
        # pool.map 会返回一个嵌套列表
        all_output_nested = pool.map(run_single_material_task, tasks)

    # 2. 【关键汇总】：将嵌套列表展平为一个扁平列表
    final_combined = [res for sublist in all_output_nested for res in sublist]

    # 3. 统一保存：在主进程中执行一次性保存
    if final_combined:
        import pandas as pd
        df = pd.DataFrame(final_combined)
        # 增加排序逻辑：先按材料名排，再按环境名排
        df = df.sort_values(by=["Material", "Environment"])
        csv_path = os.path.join(base_dir, "Final_Formation_Energies_All.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✅ 所有任务汇总完成！总计 {len(df)} 条数据。")
        print(f"📊 最终报表已生成: {csv_path}")

if __name__ == "__main__":
    main()  # 执行主函数
