import os
import sys
import glob
import numpy as np
import subprocess
import time
import re
from datetime import datetime
from ase.io import read
from ase.geometry import get_distances

# ==========================================
# 1. 用户配置区域
# ==========================================
MPI_CMD = "mpirun --allow-run-as-root --oversubscribe -np 16"

NEB_PARAMS = {
    "num_of_images": 7,
    "opt_scheme": "broyden",
    "CI_scheme": "no-CI",  # 建议先跑 no-CI，收敛后再跑 CI
    "k_max": 0.3,
    "k_min": 0.2,
    "path_thr": 0.5
}

# ==========================================
# 2. 辅助功能 (颜色与日志)
# ==========================================
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log(msg, level="info"):
    if level == "info": print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} {msg}")
    elif level == "success": print(f"{Colors.OKGREEN}[DONE]{Colors.ENDC} {msg}")
    elif level == "warn": print(f"{Colors.WARNING}[WARN]{Colors.ENDC} {msg}")
    elif level == "error": print(f"{Colors.FAIL}[ERR]{Colors.ENDC} {msg}")
    elif level == "step": print(f"\n{Colors.HEADER}=== {msg} ==={Colors.ENDC}")

# ==========================================
# 3. 核心功能类
# ==========================================

class SmartNEBRunner:
    def __init__(self):
        self.pseudo_dir = self._find_pseudo_dir()
        
    def _find_pseudo_dir(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(script_dir, "pseudos")
        if os.path.exists(target_path) and os.path.isdir(target_path):
            return target_path
        else:
            log(f"未找到 pseudos 文件夹，请检查！路径: {script_dir}", "error")
            return "./pseudos"

    def find_latest_run_dir(self):
        search_pattern = "FullRun_NEB_*"
        candidates = glob.glob(search_pattern)
        dirs = [d for d in candidates if os.path.isdir(d)]
        if not dirs: return None
        return max(dirs, key=os.path.getmtime)

    def scan_and_run(self, root_dir):
        log(f"扫描目录: {root_dir}")
        search_path = os.path.join(root_dir, "*_IS")
        is_dirs = glob.glob(search_path)
        is_dirs.sort()
        
        if not is_dirs:
            log(f"未找到 _IS 文件夹", "warn")
            return

        found_count = 0
        for dir_vac_is in is_dirs:
            out_vac_is = os.path.join(dir_vac_is, 'espresso.pwo')
            
            if not os.path.exists(out_vac_is) or not self.is_job_done(out_vac_is):
                continue

            neb_dir = os.path.join(dir_vac_is, "neb_calc")
            if self.is_neb_done(neb_dir):
                continue

            task_name = os.path.basename(dir_vac_is)
            log(f"开始处理任务: {task_name}", "info")
            self.process_single_neb(dir_vac_is, out_vac_is, neb_dir)
            found_count += 1
            
        if found_count == 0:
            log("没有新的任务需要计算。", "success")

    def is_job_done(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 2000), 0)
                return "JOB DONE" in f.read().decode('utf-8', errors='ignore')
        except:
            return False

    def is_neb_done(self, neb_dir):
        neb_outfile = os.path.join(neb_dir, "neb.out")
        if os.path.exists(neb_outfile):
            return self.parse_barrier(neb_outfile) != "N/A"
        return False

    def process_single_neb(self, is_dir, pwo_file, neb_dir):
        if not os.path.exists(neb_dir):
            os.makedirs(neb_dir)

        log(f"读取结构 -> {os.path.basename(pwo_file)}", "step")
        try:
            initial_atoms = read(pwo_file, format='espresso-out', index=-1)
        except Exception as e:
            log(f"结构读取错误: {e}", "error")
            return

        vac_pos = self.find_vacancy_grid(initial_atoms)
        jump_idx, jump_dist = self.find_jump_atom(initial_atoms, vac_pos)
        
        if jump_idx is None: return

        log(f"跳跃氧原子 Index: {jump_idx} | 距离: {jump_dist:.3f} A", "info")

        final_atoms = initial_atoms.copy()
        final_atoms.positions[jump_idx] = vac_pos
        
        inp_path = os.path.join(neb_dir, "neb.inp")
        self.write_neb_input(inp_path, initial_atoms, final_atoms)
        
        # 调用新的监控运行函数
        self.run_neb_monitor(neb_dir, "neb.inp", "neb.out")

    def find_vacancy_grid(self, atoms):
        cell = atoms.get_cell()
        positions = atoms.get_positions()
        n_grid = 10
        xs = np.linspace(0, 1, n_grid)
        grid = np.array(np.meshgrid(xs, xs, xs)).T.reshape(-1, 3)
        grid_cart = np.dot(grid, cell)
        max_dist = 0
        vac_pos = None
        for point in grid_cart:
            dists = np.linalg.norm(positions - point, axis=1)
            min_dist = np.min(dists)
            if min_dist > max_dist:
                max_dist = min_dist
                vac_pos = point
        return vac_pos

    def find_jump_atom(self, atoms, vac_pos):
        min_dist = 10.0
        jump_idx = None
        for i, atom in enumerate(atoms):
            if atom.symbol == 'O':
                d = get_distances(vac_pos, atoms.positions[i], cell=atoms.cell, pbc=atoms.pbc)
                dist_val = d[1][0][0]
                if dist_val < min_dist:
                    min_dist = dist_val
                    jump_idx = i
        return jump_idx, min_dist

    def write_neb_input(self, filename, initial, final):
        def get_pos_str(atoms):
            s = ""
            for a in atoms:
                s += f"{a.symbol} {a.position[0]:.6f} {a.position[1]:.6f} {a.position[2]:.6f} 1 1 1\n"
            return s

        elements = sorted(list(set(initial.get_chemical_symbols())))
        mass_dict = {'Zr': 91.22, 'Sc': 44.96, 'Y': 88.91, 'O': 16.00}
        pseudo_files = {
            'Zr': 'Zr.pbe-spn-kjpaw_psl.1.0.0.UPF',
            'Sc': 'Sc.pbe-spn-kjpaw_psl.1.0.0.UPF',
            'Y':  'Y.pbe-spn-kjpaw_psl.1.0.0.UPF',
            'O':  'O.pbe-n-kjpaw_psl.1.0.0.UPF'
        }

        atomic_species = ""
        for el in elements:
            mass = mass_dict.get(el, 1.0)
            p_file = pseudo_files.get(el, f"{el}.UPF")
            atomic_species += f"{el} {mass} {p_file}\n"

        content = f"""BEGIN
BEGIN_PATH_INPUT
&PATH
  restart_mode      = 'from_scratch'
  string_method     = 'neb'
  nstep_path        = 50
  ds                = 1.0D0
  opt_scheme        = '{NEB_PARAMS['opt_scheme']}'
  num_of_images     = {NEB_PARAMS['num_of_images']}
  CI_scheme         = '{NEB_PARAMS['CI_scheme']}'
  use_freezing      = .true.
  k_max             = {NEB_PARAMS['k_max']}
  k_min             = {NEB_PARAMS['k_min']}
  path_thr          = {NEB_PARAMS['path_thr']}
/
END_PATH_INPUT
BEGIN_ENGINE_INPUT
&CONTROL
  prefix     = 'calc'
  outdir     = './tmp_neb'
  pseudo_dir = '{self.pseudo_dir}'
  disk_io    = 'low'
/
&SYSTEM
  ibrav = 0, nat = {len(initial)}, ntyp = {len(elements)}
  ecutwfc     = 50
  ecutrho     = 400
  occupations = 'smearing'
  smearing    = 'gaussian'
  degauss     = 0.01
/
&ELECTRONS
  conv_thr         = 1.0e-5
  mixing_beta      = 0.3
/
ATOMIC_SPECIES
{atomic_species}
K_POINTS gamma
BEGIN_POSITIONS
FIRST_IMAGE
ATOMIC_POSITIONS (angstrom)
{get_pos_str(initial)}
LAST_IMAGE
ATOMIC_POSITIONS (angstrom)
{get_pos_str(final)}
END_POSITIONS
END_ENGINE_INPUT
END
"""
        with open(filename, 'w') as f:
            f.write(content)

    # ========================================================
    # ★ 核心修改：实时监控函数 ★
    # ========================================================
    def run_neb_monitor(self, work_dir, inp_file, out_file):
        cwd = os.getcwd()
        os.chdir(work_dir)
        
        cmd = f"{MPI_CMD} neb.x -input {inp_file}"
        log(f"启动计算监控 (Real-time Monitor)...", "info")
        
        start_t = time.time()
        
        # 打开文件准备写入日志
        with open(out_file, "w") as f_out:
            # 启动子进程，并将 stdout 导向 PIPE 以便 Python 读取
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, # 将错误也合并到输出流
                text=True, # 以文本形式读取，而不是字节
                bufsize=1  # 行缓冲
            )
            
            # 实时读取循环
            step_counter = 0
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                if line:
                    # 1. 写入文件 (由 Python 负责写日志文件)
                    f_out.write(line)
                    f_out.flush() # 确保立即写入磁盘

                    # 2. 解析进度并打印到屏幕
                    # 捕获 "iteration" 步数
                    if "neb: iteration" in line:
                        step_counter += 1
                        # 简单的 print 会换行，用 \r 可以覆盖同一行实现动画效果
                        print(f"\r  >>> 正在进行第 {Colors.BOLD}{step_counter}{Colors.ENDC} 步优化...", end="")
                    
                    # 捕获 "activation energy"
                    if "activation energy (->)" in line:
                        try:
                            # 格式: activation energy (->) =   0.8520 eV
                            parts = line.split('=')
                            e_act = float(parts[1].split()[0])
                            # 打印能垒信息
                            print(f"\r  >>> [Step {step_counter}] 当前能垒: {Colors.OKGREEN}{e_act:.4f} eV{Colors.ENDC}      ", end="\n")
                        except:
                            pass
                    
                    # 捕获错误
                    if "Error" in line or "error" in line:
                        if "path_thr" not in line: # 忽略参数打印里的 error 单词
                            print(f"\n  {Colors.FAIL}[QE Error] {line.strip()}{Colors.ENDC}")

        # 等待进程完全结束
        process.wait()
        duration = time.time() - start_t
        os.chdir(cwd)

        if process.returncode == 0:
            em = self.parse_barrier(os.path.join(work_dir, out_file))
            log(f"计算成功 (耗时 {duration:.1f}s) | ★ 最终能垒 Em = {em} eV", "success")
        else:
            log(f"计算异常终止 (耗时 {duration:.1f}s)", "error")

    def parse_barrier(self, outfile):
        try:
            val = "N/A"
            with open(outfile, 'r') as f:
                for line in f:
                    if "activation energy (->)" in line:
                        val = line.split('=')[1].split()[0]
            return val
        except:
            return "N/A"

if __name__ == "__main__":
    runner = SmartNEBRunner()
    
    print(f"{Colors.BOLD}=== 氧离子迁移能垒计算 (实时监控) ==={Colors.ENDC}")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    target_dir = runner.find_latest_run_dir()
    
    if target_dir:
        print(f"检测到最新的 Ef 计算目录: {Colors.OKCYAN}{target_dir}{Colors.ENDC}")
        val = input("是否运行? (y/n): ")
        if val.lower() == 'y':
            runner.scan_and_run(target_dir)
    else:
        print(f"{Colors.WARNING}未找到 FullRun_NEB_* 文件夹。{Colors.ENDC}")
        path = input("请输入路径: ")
        if os.path.exists(path):
            runner.scan_and_run(path)