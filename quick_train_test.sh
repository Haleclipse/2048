#!/bin/bash

# 快速训练测试 - 验证TD学习和分阶段训练流程

echo "=== 快速训练测试 ==="
echo "验证TD学习和分阶段训练功能"
echo

# 阶段1: 基础学习 (100局)
echo "阶段1: 基础学习 (100局)"
./2048 --total=100 --slide="init=1000,1000,1000,1000 alpha=0.1 learning=1 penalty=0.0 bonus=100" --save=test_stage1.w

echo
echo "阶段2: 危险感知 (100局)" 
./2048 --total=100 --slide="load=test_stage1.w alpha=0.05 learning=1 penalty=0.5 bonus=300" --save=test_stage2.w

echo
echo "阶段3: 精细调整 (100局)"
./2048 --total=100 --slide="load=test_stage2.w alpha=0.01 learning=1 penalty=0.8 bonus=500" --save=test_final.w

echo
echo "最终测试 (50局，学习关闭)"
./2048 --total=50 --slide="load=test_final.w alpha=0 learning=0"

echo
echo "快速训练测试完成！"
echo "生成的权重文件: test_stage1.w, test_stage2.w, test_final.w"