import os
import glob
from ase.io import read
from ase.neb import NEB

def build_qe_neb_input(is_file, fs_out_file, n_images=5, output_file='neb.in'):
    """读取初态和终态，进行 MIC 插值并生成 neb.in"""
    
    initial = read(is_file, format='espresso-in')
    final = read(fs_out_file, format='espresso-out')

    # 1. 构建插值列表
    images = [initial]
    for _ in range(n_images):
        images.append(initial.copy())
    images.append(final)

    # 2. ASE 线性插值（mic=True 解决跨越周期性边界问题）
    neb = NEB(images)
    neb.interpolate(mic=True)

    # 3. 提取元素符号和参数准备写入
    symbols = initial.get_chemical_symbols()
    
    # ⚠️注意：这部分的参数 (ecutwfc, ecutrho, K点等) 必须和你跑 relax 时绝对一致！
    engine_input = """&CONTROL
   calculation      = 'neb'
   restart_mode     = 'from_scratch'
   outdir           = './tmp_neb'
   prefix           = 'calc_NEB'
   pseudo_dir       = '../pseudos'
/
&SYSTEM
   ecutwfc          = 60
   ecutrho          = 480
   occupations      = 'smearing'
   degauss          = 0.005
   smearing         = 'gaussian'
   ntyp             = 4
   nat              = 95
   ibrav            = 0
/
&ELECTRONS
   electron_maxstep = 100
   conv_thr         = 1e-06
   mixing_beta      = 0.3
/
&IONS
/
ATOMIC_SPECIES
Zr 91.224   Zr.pbe-spn-kjpaw_psl.1.0.0.UPF
O  15.999   O.pbe-n-kjpaw_psl.1.0.0.UPF
Sc 44.9559  Sc.pbe-spn-kjpaw_psl.1.0.0.UPF
Y  88.9058  Y.pbe-spn-kjpaw_psl.1.0.0.UPF

K_POINTS automatic
2 2 2  0 0 0

CELL_PARAMETERS angstrom
10.25000000000000 0.00000000000000 0.00000000000000
0.00000000000000 10.25000000000000 0.00000000000000
0.00000000000000 0.00000000000000 10.25000000000000
"""

    path_input = f"""BEGIN_PATH_INPUT
&PATH
  restart_mode  = 'from_scratch'
  string_method = 'neb'
  nstep_path    = 100
  ds            = 1.0
  opt_scheme    = 'broyden'
  num_of_images = {n_images + 2}
  k_max         = 0.3
  k_min         = 0.2
  CI_scheme     = 'auto'
  path_thr      = 0.05
/
END_PATH_INPUT
"""

    with open(output_file, 'w') as f:
        f.write(path_input)
        f.write("BEGIN_ENGINE_INPUT\n")
        f.write(engine_input)
        f.write("BEGIN_POSITIONS\n")
        
        # 初态
        f.write("FIRST_IMAGE\n")
        f.write("ATOMIC_POSITIONS (angstrom)\n")
        for sym, pos in zip(symbols, images[0].positions):
            f.write(f"{sym:<4} {pos[0]:15.10f} {pos[1]:15.10f} {pos[2]:15.10f}\n")
            
        # 中间态
        for i in range(1, n_images + 1):
            f.write(f"INTERMEDIATE_IMAGE\n")
            f.write("ATOMIC_POSITIONS (angstrom)\n")
            for sym, pos in zip(symbols, images[i].positions):
                f.write(f"{sym:<4} {pos[0]:15.10f} {pos[1]:15.10f} {pos[2]:15.10f}\n")
                
        # 终态
        f.write("LAST_IMAGE\n")
        f.write("ATOMIC_POSITIONS (angstrom)\n")
        for sym, pos in zip(symbols, images[-1].positions):
            f.write(f"{sym:<4} {pos[0]:15.10f} {pos[1]:15.10f} {pos[2]:15.10f}\n")
            
        f.write("END_POSITIONS\n")
        f.write("END_ENGINE_INPUT\n")

if __name__ == "__main__":
    base_dir = os.getcwd()
    target_dirs = sorted(glob.glob("*_Vac_4Zr"))
    
    for dir_name in target_dirs:
        if os.path.isdir(dir_name):
            print(f"[*] 正在处理体系: {dir_name}")
            os.chdir(dir_name)
            
            # 检查必要文件是否存在
            if os.path.exists("IS.in") and os.path.exists("FS_relax.out"):
                try:
                    build_qe_neb_input("IS.in", "FS.out", n_images=5, output_file="neb.in")
                    print("    ✅ neb.in 生成成功！")
                except Exception as e:
                    print(f"    ❌ 生成失败，错误信息: {e}")
            else:
                print("    ⚠️ 缺失 IS.in 或 FS_relax.out，跳过。")
            
            os.chdir(base_dir)
    print("🎉 所有 NEB 输入文件生成完毕！")
