#!/usr/bin/env python3
"""
测试独立进度条功能
"""
import numpy as np
import sys
import time
from sus_failure_probability import subset_simulation, run_subset_simulation
import multiprocessing

def test_independent_progress_bars():
    """测试每个CPU核心的独立进度条"""
    print("测试独立进度条功能")
    print("=" * 60)
    
    # 启用进度条
    import sus_failure_probability
    sus_failure_probability.SHOW_PROGRESS = True
    
    # 加载预计算结果
    try:
        res_no_adpt = np.load("res_no_adpt.npy", allow_pickle=True).item()
        print("✓ 成功加载预计算结果")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return False
    
    # 创建测试参数 - 使用小样本快速测试
    N = 100  # 小样本数量
    p0 = 0.1
    k_mean = 1.0
    n_test_runs = 4  # 测试4个并行任务
    
    print(f"\n准备测试参数:")
    print(f"  样本数量: {N}")
    print(f"  并行任务数: {n_test_runs}")
    print(f"  CPU核心数: {multiprocessing.cpu_count()}")
    
    # 准备参数，每个任务有唯一的进程ID
    test_args = []
    for i in range(n_test_runs):
        seed = np.random.randint(0, 100000)
        process_id = i  # 给每个任务分配唯一ID
        args = (N, p0, res_no_adpt['u'], res_no_adpt['time_grid'], k_mean, seed, process_id)
        test_args.append(args)
    
    print(f"\n开始并行执行 {len(test_args)} 个任务...")
    print("每个CPU核心应该显示独立的进度条\n")
    
    start_time = time.time()
    
    # 使用多进程执行
    with multiprocessing.Pool(processes=min(n_test_runs, multiprocessing.cpu_count())) as pool:
        results = pool.map(run_subset_simulation, test_args)
    
    end_time = time.time()
    
    print(f"\n测试完成!")
    print(f"执行时间: {end_time - start_time:.2f}秒")
    print(f"结果:")
    for i, result in enumerate(results):
        print(f"  任务{i+1}: 失效概率={result:.6f}")
    
    # 验证结果
    all_finite = all(np.isfinite(r) and 0 <= r <= 1 for r in results)
    if all_finite:
        print("✓ 所有结果都是有效的失效概率")
        return True
    else:
        print("✗ 部分结果无效")
        return False

def test_single_process():
    """测试单进程模式"""
    print("\n\n单进程测试（对比）")
    print("=" * 40)
    
    import sus_failure_probability
    sus_failure_probability.SHOW_PROGRESS = True
    
    try:
        res_no_adpt = np.load("res_no_adpt.npy", allow_pickle=True).item()
        
        N = 50
        p0 = 0.1
        k_mean = 1.0
        seed = 12345
        process_id = 0
        
        print(f"单进程执行 (N={N})...")
        start_time = time.time()
        result = subset_simulation(N, p0, res_no_adpt['u'], res_no_adpt['time_grid'], k_mean, seed, process_id)
        end_time = time.time()
        
        print(f"单进程结果: {result:.6f}")
        print(f"执行时间: {end_time - start_time:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"单进程测试失败: {e}")
        return False

if __name__ == "__main__":
    print("独立进度条功能测试")
    print("=" * 70)
    
    success1 = test_independent_progress_bars()
    success2 = test_single_process()
    
    print("\n" + "=" * 70)
    if success1 and success2:
        print("✓ 所有测试通过！独立进度条功能正常工作")
    else:
        print("⚠ 部分测试失败")
    print("=" * 70)
