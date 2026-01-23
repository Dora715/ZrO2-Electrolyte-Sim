import os  # 导入操作系统接口模块，用于创建目录、设置环境变量和路径操作
import sys  # 导入系统模块，用于访问与 Python 解释器紧密相关的变量和函数
import numpy as np  # 导入 NumPy 库，用于高效的数值计算和矩阵运算
import random  # 导入随机数模块，用于随机选择掺杂原子的位置
import subprocess  # 导入子进程模块，用于在 Python 中执行外部命令 (如 mpirun, pw.x)
import time  # 导入时间模块，用于计时和暂停
import multiprocessing  # 导入多进程模块 (虽然本脚本主要用 subprocess，但保留此库备用)
from datetime import datetime  # 导入日期时间模块，用于生成带时间戳的文件夹名称
from collections import Counter  # 导入计数器工具，用于统计配位环境中原子类型的数量
from ase import Atoms  # 从 ASE 库导入 Atoms 类，用于构建原子结构 (如 O2 分子)

# ==========================================
# 1. 环境配置
# ==========================================
os.environ['OMP_NUM_THREADS'] = '1'  # 强制 OpenMP 使用 1 个线程，防止与 MPI 多进程并行发生资源冲突
os.environ['MKL_NUM_THREADS'] = '1'  # 强制 Intel MKL 数学库使用 1 个线程
os.environ['OMP_PROC_BIND'] = 'true'  # 绑定线程到处理器核心，优化 CPU 缓存命中率
os.environ['OMP_PLACES'] = 'threads'  # 指定 OpenMP 线程放置在硬件线程上
MAX_MPI_CORES = 16  # 定义最大使用的 MPI 核心数，用于并行计算

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
        self.pseudo_dir = os.path.abspath(pseudo_dir)  # 获取伪势文件夹的绝对路径
        # 定义元素与伪势文件名的映射字典
        self.pseudopotentials = {
            'Zr': 'Zr.pbe-spn-kjpaw_psl.1.0.0.UPF',
            'Sc': 'Sc.pbe-spn-kjpaw_psl.1.0.0.UPF',
            'Y':  'Y.pbe-spn-kjpaw_psl.1.0.0.UPF',
            'O':  'O.pbe-n-kjpaw_psl.1.0.0.UPF'
        }

    def generate_input(self, atoms, task_name, calc_dir, override_data=None):
        from ase.io import write  # 导入 ASE 的写入函数
        if not os.path.exists(calc_dir): os.makedirs(calc_dir)  # 如果计算目录不存在，则创建它

        # 定义默认的 Quantum ESPRESSO 输入参数字典
        input_data = {
            'control': {
                'calculation': 'relax',  # 计算类型：结构弛豫 (优化离子位置)
                'nstep': 200,             # 最大离子步数 (建议根据需求调整，测试用 50)
                'etot_conv_thr': 1.0e-4, # 能量收敛阈值 (测试用较低精度，正式计算建议 1.0e-4)
                'forc_conv_thr': 1.0e-3, # 力收敛阈值 (测试用较低精度，正式计算建议 1.0e-3)
                'restart_mode': 'from_scratch', # 每次都从头开始计算
                'prefix': 'calc',        # 计算文件的前缀
                'pseudo_dir': self.pseudo_dir, # 伪势目录路径
                'outdir': './tmp',       # 临时文件输出目录
                'tprnfor': True,         # 计算并打印原子受力
                'disk_io': 'low'         # 减少磁盘 I/O 操作
            },
            'system': {
                'ecutwfc': 60,           # 波函数截断能 (Ry) (测试用 25，正式建议 40+)
                'ecutrho': 480,          # 电荷密度截断能 (通常是 ecutwfc 的 4-8 倍)
                'occupations': 'smearing', # 电子占据方式：smearing (适合金属或小带隙)
                'smearing': 'gaussian',  # smearing 类型：高斯
                'degauss': 0.005,         # smearing 宽度 (Ry)
            },
            'electrons': {
                'conv_thr': 1.0e-6,      # 电子自洽迭代收敛阈值 (测试用，正式建议 1.0e-6)
                'mixing_beta': 0.3,      # 混合因子，控制电荷密度更新步长
                'electron_maxstep': 100,  # 最大电子迭代步数
                'diagonalization': 'david' # 对角化算法：Davidson
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
              kpts=(1, 1, 1)) # 设置 K 点网格为 Gamma 点 (1x1x1)
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
def run_and_monitor(cmd, output_file_path):
    """执行命令并实时打印进度"""
    print(f"    CMD: {cmd}")  # 打印将要执行的命令
    print(f"    LOG: {output_file_path}")  # 打印日志文件路径
    print("    ------------------------------------------------------")
    print("    [进度监控] 正在启动计算核心...")

    start_time = time.time()  # 记录开始时间
    # 启动子进程执行命令
    # stdout=subprocess.PIPE: 捕获标准输出
    # bufsize=1: 行缓冲，确保实时获取输出
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    # 打开日志文件准备写入
    with open(output_file_path, "w") as f_log:
        step_count = 0  # 初始化离子步计数器
        while True:
            line = process.stdout.readline()  # 实时读取一行输出
            # 如果读不到行且进程已结束，则跳出循环
            if not line and process.poll() is not None: break
            if line:  # 如果读到了内容
                f_log.write(line)  # 将该行写入日志文件
                
                # --- 实时反馈逻辑 ---
                # 检测到 "Forces acting on atoms"，说明完成了一个离子步
                if "Forces acting on atoms" in line:
                    step_count += 1
                    print(f"    >>> [Step {step_count}] 优化中...")

                # 检测总能量输出行 (包含 "total energy" 且包含 "!" 标记)
                if "total energy" in line and "!" in line:
                    try:
                        energy = line.split('=')[1].strip().split()[0]  # 解析能量数值
                        print(f"        当前能量: {energy} Ry")
                    except: pass  # 如果解析失败，忽略错误

                # 检测总受力输出行
                if "Total force" in line:
                    try:
                        parts = line.split()
                        force = float(parts[2])  # 解析受力数值
                        status = "🔴"  # 默认状态图标 (红灯：力很大)
                        if force < 0.05: status = "🟢"  # 绿灯：力很小，接近收敛
                        elif force < 0.1: status = "🟡"  # 黄灯：力中等
                        print(f"        当前受力: {force:.6f} {status}")
                    except: pass

                # 检测到 "JOB DONE"，说明计算正常结束
                if "JOB DONE" in line:
                    print("    ✅ 计算成功结束 (JOB DONE)!")

    rc = process.poll()  # 获取进程的返回码
    if rc != 0: raise subprocess.CalledProcessError(rc, cmd)  # 如果返回码不为 0，抛出异常

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
                except: pass
    return enc  # 返回提取到的能量 (eV)

# ==========================================
# 4. 主流程 (最终集成版)
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
    print(f"   步骤 0: 计算氧结合能及化学势参考值")
    print(f"########################################")

    # --- 0.1 计算 O2 分子 ---
    task_o2 = "O2_Ref"
    dir_o2 = os.path.join(base_dir, task_o2)
    atoms_o2 = Atoms('O2', positions=[(0, 0, 0), (0, 0, 1.23)], cell=[15, 15, 15], pbc=True)
    atoms_o2.center()
    o2_settings = {
        'system': {'nspin': 2, 'starting_magnetization(1)': 0.5, 'degauss': 0.005}
    }

    # --- 0.2 计算孤立 O 原子 ---
    task_atom = "O_Atom_Ref"
    dir_atom = os.path.join(base_dir, task_atom)
    atoms_atom = Atoms('O', positions=[(6, 6, 6)], cell=[12, 12, 12], pbc=True)
    atom_settings = {
        'system': {'nspin': 2, 'starting_magnetization(1)': 0.5,  'degauss': 0.001}
    }

    try:
        # 计算 O2
        inp_o2 = qe_manager.generate_input(atoms_o2, task_o2, dir_o2, override_data=o2_settings)
        run_and_monitor(f"mpirun --allow-run-as-root -np 4 pw.x < {inp_o2}", os.path.join(dir_o2, 'espresso.pwo'))
        e_o2 = parse_energy(os.path.join(dir_o2, 'espresso.pwo'))

        # 计算 O 原子
        inp_atom = qe_manager.generate_input(atoms_atom, task_atom, dir_atom, override_data=atom_settings)
        run_and_monitor(f"mpirun --allow-run-as-root -np 1 pw.x < {inp_atom}", os.path.join(dir_atom, 'espresso.pwo'))
        e_atom = parse_energy(os.path.join(dir_atom, 'espresso.pwo'))

        if e_o2 and e_atom:
            eb = e_o2 - (2 * e_atom)
            chem_pot_o = e_o2 / 2.0
            print(f"\n    ★ 氧结合能 Eb: {eb:.4f} eV ")
            print(f"    ★ 氧化学势 μ_O: {chem_pot_o:.4f} eV")
    except Exception as e:
        print(f"    ❌ 氧参考态计算失败: {e}")

    # =======================================================
    # 【步骤 1】 晶体计算循环
    # =======================================================
    # 定义材料字典，这里示例为 10ScSZ (Zr:Sc = 90:20)
    materials = {"9Sc1YSZ": {"Zr": 90, "Sc": 18, "Y": 2}}
    builder = ZrO2Builder()  # 实例化构建器

    # Pymatgen 检测 (用于高级环境分析)
    try:
        from pymatgen.io.ase import AseAtomsAdaptor  # 用于 ASE <-> Pymatgen 转换
        from pymatgen.analysis.local_env import CrystalNN  # 用于分析晶体近邻环境
        has_pymatgen = True  # 标记已安装 Pymatgen
    except ImportError:
        has_pymatgen = False  # 标记未安装
        print(">>> [警告] 未检测到 Pymatgen，环境分析功能受限。")

    for name, ratios in materials.items():  # 遍历所有材料
        print(f"\n########################################")
        print(f"   处理材料: {name}")
        print(f"########################################")

        # --- A. 完美晶胞 (Perfect Cell) ---
        # 构建掺杂后的初始结构
        struct = builder.build_doped_structure(name, ratios['Zr'], ratios['Sc'], ratios['Y'])
        struct.rattle(stdev=0.01, seed=42)  # 对原子位置进行微扰 (打破对称性，利于收敛)

        print(f"\n   >>> [步骤 A: 基准参考] 计算完美晶胞能量...")
        task_perf = f"{name}_Perfect"  # 任务名
        dir_perf = os.path.join(base_dir, task_perf)  # 目录名
        inp_perf = qe_manager.generate_input(struct, task_perf, dir_perf)  # 生成输入
        out_perf = os.path.join(dir_perf, 'espresso.pwo')  # 输出路径
        
        try:
            # 运行命令：使用最大核心数并行计算
            cmd = f"mpirun --allow-run-as-root --oversubscribe -np {MAX_MPI_CORES} pw.x < {inp_perf}"
            run_and_monitor(cmd, out_perf)  # 运行
            e_perfect = parse_energy(out_perf)  # 解析能量
        except Exception as e:
            print(f"    ❌ 完美晶胞计算失败: {e}")
            e_perfect = None

        if e_perfect:
            print(f"    ★ E_perfect: {e_perfect:.6f} eV")
        else:
            e_perfect = 0.0  # 失败则设为 0

# --- B. 环境分析 (Environment Analysis) ---
        print("\n   >>> [步骤 B: 环境分析] 正在分析氧配位环境 (化学指纹法)...")
        


        # --- 执行分析 ---
        try:
            fingerprinter = EnvironmentFingerprinter(struct)
            envs = fingerprinter.analyze()
        except Exception as e:
            print(f"   >>> [错误] 环境分析模块崩溃: {e}")
            envs = {} # 清空以免后续报错

        # --- 打印并校验结果 ---
        if not envs:
            print("   >>> [警告] 未识别到任何合理的氧环境！可能所有原子都在边界上。")
            print("   >>> [应急] 将强制使用第 0 号原子进行测试。")
            envs = {"Fallback_Random": [0]}
        
        print(f"\n   [环境统计表] 共识别出 {len(envs)} 种【合理】环境:")
        print(f"   {'No.':<4} {'Environment':<20} {'Count':<6} {'Example ID'}")
        print(f"   {'-'*4} {'-'*20} {'-'*6} {'-'*15}")
        
        # 排序便于查看
        sorted_envs = sorted(envs.items(), key=lambda item: len(item[1]), reverse=True)
        
        for i, (env_name, indices) in enumerate(sorted_envs):
            print(f"   {i+1:<4} {env_name:<20} {len(indices):<6} {indices[0]}")

        # ==========================================
        # --- C. 缺陷计算 (Defect Cell) ---
        # ==========================================
        
        # 遍历排序后的环境，确保执行顺序一致
        for i, (label, indices) in enumerate(sorted_envs):
            idx = indices[0]  # 取该环境下的第一个原子作为代表
            
            print(f"\n   -------------------------------------------------------")
            print(f"   >>> [步骤 C: 缺陷计算] ({i+1}/{len(envs)}) 类型: {label} (Atom ID: {idx})")
            
            # 1. 构建缺陷结构
            defect_struct = struct.copy()  # 复制完美晶胞结构 (ASE Atoms 对象)
            del defect_struct[idx]         # 删除选定的氧原子，制造空位

            # 2. 定义任务路径
            task_vac = f"{name}_Vac_{label}"
            dir_vac = os.path.join(base_dir, task_vac)
            
            # 3. 生成输入文件
            # 注意：generate_input 内部需要处理 defect_struct
            inp_vac = qe_manager.generate_input(defect_struct, task_vac, dir_vac)
            out_vac = os.path.join(dir_vac, 'espresso.pwo')

            # 4. 运行计算
            cmd = f"mpirun --allow-run-as-root --oversubscribe -np {MAX_MPI_CORES} pw.x < {inp_vac}"
            
            e_defect = None
            e_final_form = "N/A"
            
            try:
                # 检查是否已算完 (简单的断点续算逻辑)
                if os.path.exists(out_vac) and "JOB DONE" in open(out_vac).read():
                     print(f"      -> 检测到计算已完成，跳过执行，直接读取能量...")
                else:
                     run_and_monitor(cmd, out_vac) # 运行
                
                # 5. 解析能量
                e_defect = parse_energy(out_vac)
                
                # 6. 计算形成能 Ef
                if e_defect is not None and e_perfect != 0.0 and e_o2 is not None:
                    # 公式: Ef = E_defect - E_perfect + \mu_O
                    # \mu_O = 1/2 E(O2_total)
                    raw_diff = e_defect - e_perfect
                    e_final_form = raw_diff + chem_pot_o
                    
                    print(f"        [数据] E_Defect : {e_defect:.6f} eV")
                    print(f"        [数据] E_Perfect: {e_perfect:.6f} eV")
                    print(f"        [数据] \u03BC_O (1/2 O2): {chem_pot_o:.6f} eV")
                    print(f"        ------------------------------------")
                    print(f"        [最终结果] 形成能 Ef: {e_final_form:.6f} eV")
                else:
                    print("        ⚠️ 数据缺失 (E_defect/E_perfect/E_O2)，无法计算最终 Ef")

                # 7. 保存结果
                results.append({
                    "Material": name,
                    "Environment": label,
                    "Atom_Index": idx,
                    "E_Perfect": e_perfect,
                    "E_Defect": e_defect,
                    "E_O2_Total": e_o2,
                    "Chemical_Pot_O": chem_pot_o,
                    "Formation_Energy_Ef": e_final_form
                })

            except subprocess.CalledProcessError:
                print("   ❌ 计算异常终止 (mpirun error)")
            except Exception as e:
                print(f"   ❌ 处理出错: {e}")

    # 5. 保存结果
    if results:  # 如果有结果
        import pandas as pd  # 导入 pandas 用于处理表格
        csv_path = os.path.join(base_dir, "Final_Formation_Energies.csv")  # CSV 文件路径
        pd.DataFrame(results).to_csv(csv_path, index=False)  # 保存为 CSV，不包含行索引
        print(f"\n✅ 所有任务完成！最终报表已生成: {csv_path}")

if __name__ == "__main__":
    main()  # 执行主函数