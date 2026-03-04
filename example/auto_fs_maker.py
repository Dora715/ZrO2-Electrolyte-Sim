import os

def extract_final_coords(pwo_file):
    """从 espresso.pwo 中提取最终弛豫的坐标块"""
    coords = []
    capture = False
    try:
        with open(pwo_file, 'r') as f:
            for line in f:
                if 'Begin final coordinates' in line:
                    capture = True
                    coords = [] # 清空之前的记录，只保留最后一次
                    continue
                if 'End final coordinates' in line:
                    capture = False
                    continue
                if capture:
                    if line.strip() == '': continue
                    coords.append(line)
        return coords
    except FileNotFoundError:
        print(f"  [跳过] 找不到文件: {pwo_file}")
        return None

def process_folder(folder_path, oa_index, vac_xyz):
    """处理单个文件夹：读取 pwi 模板，注入最终坐标生成 IS.in，替换坐标生成 FS.in"""
    print(f"\n[{folder_path}] 正在处理...")
    
    pwo_file = os.path.join(folder_path, 'espresso.pwo')
    pwi_file = os.path.join(folder_path, 'espresso.pwi')
    is_file = os.path.join(folder_path, 'IS.in')
    fs_file = os.path.join(folder_path, 'FS.in')

    # 1. 提取最终弛豫坐标
    final_coords = extract_final_coords(pwo_file)
    if not final_coords:
        return

    # 2. 读取原始输入文件 (pwi) 作为模板，剔除旧的坐标块
    preamble = []
    postamble = []
    mode = 'preamble'
    
    try:
        with open(pwi_file, 'r') as f:
            for line in f:
                if 'ATOMIC_POSITIONS' in line.upper():
                    mode = 'skip_coords'
                    continue
                # 如果在坐标块之后还有 K_POINTS 等其他卡片，则切换到 postamble 模式
                if mode == 'skip_coords' and any(card in line.upper() for card in ['K_POINTS', 'CELL_PARAMETERS']):
                    mode = 'postamble'
                
                if mode == 'preamble':
                    preamble.append(line)
                elif mode == 'postamble':
                    postamble.append(line)
    except FileNotFoundError:
        print(f"  [错误] 找不到模板文件: {pwi_file}")
        return

    # 3. 生成初态 (IS.in)
    with open(is_file, 'w') as f:
        f.writelines(preamble)
        f.writelines(final_coords)  # 注入 pwo 提取的最终坐标
        f.writelines(postamble)
    print(f"  -> 生成成功: IS.in (直接复用了原缺陷晶胞的弛豫结果)")

    # 4. 构造终态坐标 (平移 O_A)
    fs_coords = []
    atom_count = 0
    for line in final_coords:
        parts = line.split()
        if len(parts) >= 4 and parts[0].isalpha():
            atom_count += 1
            if atom_count == oa_index:
                elem = parts[0]
                # 替换为目标空位坐标
                line = f"{elem:<4} {vac_xyz[0]:>15.10f} {vac_xyz[1]:>15.10f} {vac_xyz[2]:>15.10f}\n"
                print(f"  -> 原子移动: 第 {oa_index} 号 {elem} 移至 {vac_xyz}")
        fs_coords.append(line)

    # 5. 生成终态 (FS.in)
    with open(fs_file, 'w') as f:
        f.writelines(preamble)
        f.writelines(fs_coords)
        f.writelines(postamble)
    print(f"  -> 生成成功: FS.in")

# ==========================================
# 用户自定义配置区
# ==========================================
if __name__ == "__main__":
    # 配置字典：'文件夹名称': [准备跃迁的氧原子序号(O_A), [空位X, 空位Y, 空位Z]]
    # 注意：你需要用 VESTA 打开每个体系最后的坐标，数一下 O_A 是第几个原子，并记录空位原本在哪。
    system_config = {
        '10ScSZ_Vac_4Zr':  [5, [1.27458304605, 1.2761881884, 1.277700146125]], 
        '9Sc1YSZ_Vac_4Zr': [52, [3.845841493725, 3.8381095881750005, 3.83968131515]], 
        '8Sc2YSZ_Vac_4Zr': [5, [1.2780646704, 1.270707226875, 1.2869609183750002]],
        '7Sc3YSZ_Vac_4Zr': [49, [3.8548373659000004, 1.3241443336750003, 1.301467345775]],
        '6Sc4YSZ_Vac_4Zr': [5, [1.2987243068, 1.2808919687250002, 1.26309156195]],
        '5Sc5YSZ_Vac_4Zr':[5, [1.283399890625, 1.3052946348249999, 1.3072387867750002]]
    }

    base_dir = "." # 当前执行目录 (即 ~/autodl-tmp/Em/4Zr_data)
    
    for folder_name, config in system_config.items():
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            oa_idx = config[0]
            vac_coords = config[1]
            process_folder(folder_path, oa_idx, vac_coords)
        else:
            print(f"\n[跳过] 文件夹不存在: {folder_path}")
            
    print("\n🎉 全部处理完毕！你的 IS.in 和 FS.in 已静静躺在各个文件夹里了。")
