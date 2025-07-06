#!/bin/bash

# 2048游戏特殊规则训练脚本
# 目标：训练AI在"两个8192即胜利"规则下获得高分且避免胜利

echo "开始2048特殊规则多阶段训练..."
echo "目标：避免胜利的高分AI"
echo

# 检查编译状态
if [ ! -f ./2048 ]; then
    echo "编译项目..."
    make
    if [ $? -ne 0 ]; then
        echo "编译失败，退出"
        exit 1
    fi
fi

# 创建训练日志目录
mkdir -p training_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="training_logs/training_$TIMESTAMP"
mkdir -p "$LOG_DIR"

echo "训练日志将保存到: $LOG_DIR"
echo

# 阶段1: 基础策略学习 (1万局)
echo "=== 阶段1: 基础策略学习 ==="
echo "目标: 学习基本2048策略，稳定达到2048-4096"
echo "参数: alpha=0.1, 标准奖励参数"

./2048 --total=10000 --block=1000 --limit=1000 \
       --slide="init=65536,65536,65536,65536 alpha=0.1 lambda=0.9 learning=1 penalty=0.7 bonus=1000 save=$LOG_DIR/stage1.w" \
       --save="$LOG_DIR/stage1_stats.log" \
       2>&1 | tee "$LOG_DIR/stage1_training.log"

echo "阶段1完成，权重已保存到 stage1.w"

# 验证权重文件
if [ -f "$LOG_DIR/stage1.w" ] && [ "$(file "$LOG_DIR/stage1.w" | grep -c 'data')" -gt 0 ]; then
    echo "✓ 权重文件验证通过 ($(ls -lh "$LOG_DIR/stage1.w" | awk '{print $5}'))"
else
    echo "✗ 权重文件验证失败，请检查"
    exit 1
fi
echo

# 阶段2: 危险感知训练 (8千局) 
echo "=== 阶段2: 危险感知训练 ==="
echo "目标: 学习识别危险状态，开始避免行为"
echo "参数: alpha=0.05, 中等危险惩罚"

./2048 --total=8000 --block=1000 --limit=1000 \
       --slide="load=$LOG_DIR/stage1.w alpha=0.05 lambda=0.9 learning=1 penalty=0.5 bonus=500 save=$LOG_DIR/stage2.w" \
       --save="$LOG_DIR/stage2_stats.log" \
       2>&1 | tee "$LOG_DIR/stage2_training.log"

echo "阶段2完成，权重已保存到 stage2.w"

# 验证权重文件
if [ -f "$LOG_DIR/stage2.w" ] && [ "$(file "$LOG_DIR/stage2.w" | grep -c 'data')" -gt 0 ]; then
    echo "✓ 权重文件验证通过 ($(ls -lh "$LOG_DIR/stage2.w" | awk '{print $5}'))"
else
    echo "✗ 权重文件验证失败，请检查"
    exit 1
fi
echo

# 阶段3: 精细策略调整 (5千局)
echo "=== 阶段3: 精细策略调整 ==="
echo "目标: 精确控制，最大化存活时间和分数"
echo "参数: alpha=0.01, 强化危险惩罚"

./2048 --total=5000 --block=1000 --limit=1000 \
       --slide="load=$LOG_DIR/stage2.w alpha=0.01 lambda=0.9 learning=1 penalty=0.8 bonus=300 save=$LOG_DIR/stage3.w" \
       --save="$LOG_DIR/stage3_stats.log" \
       2>&1 | tee "$LOG_DIR/stage3_training.log"

echo "阶段3完成，权重已保存到 stage3.w"

# 验证权重文件
if [ -f "$LOG_DIR/stage3.w" ] && [ "$(file "$LOG_DIR/stage3.w" | grep -c 'data')" -gt 0 ]; then
    echo "✓ 权重文件验证通过 ($(ls -lh "$LOG_DIR/stage3.w" | awk '{print $5}'))"
else
    echo "✗ 权重文件验证失败，请检查"
    exit 1
fi
echo

# 测试阶段: 评估最终性能
echo "=== 测试阶段: 性能评估 ==="
echo "运行1000局测试，学习关闭"

./2048 --total=1000 --block=100 \
       --slide="load=$LOG_DIR/stage3.w alpha=0 learning=0" \
       --save="$LOG_DIR/final_test_stats.log" \
       2>&1 | tee "$LOG_DIR/final_test.log"

echo
echo "=== 训练完成 ==="
echo "所有训练日志和权重文件保存在: $LOG_DIR"
echo

# 生成训练报告
echo "生成训练报告..."
cat > "$LOG_DIR/training_report.md" << EOF
# 2048特殊规则训练报告

训练时间: $(date)
训练目标: 避免胜利的高分AI (两个8192即胜利)

## 训练阶段

### 阶段1: 基础策略学习
- 训练局数: 10,000局
- 学习率: 0.1
- 危险惩罚: 0.7 (标准)
- 存活奖励: 1000
- 目标: 学习基本2048策略

### 阶段2: 危险感知训练  
- 训练局数: 8,000局
- 学习率: 0.05
- 危险惩罚: 0.5 (中等)
- 存活奖励: 500
- 目标: 学习识别危险状态

### 阶段3: 精细策略调整
- 训练局数: 5,000局
- 学习率: 0.01
- 危险惩罚: 0.8 (强化)
- 存活奖励: 300
- 目标: 精确控制和优化

### 最终测试
- 测试局数: 1,000局
- 学习关闭: alpha=0
- 性能评估: 见 final_test.log

## 文件说明
- stage1.w, stage2.w, stage3.w: 各阶段训练权重
- stage*_training.log: 各阶段训练日志
- final_test.log: 最终性能测试结果
- normal_games.log: 详细游戏记录
- win_games.log: 胜利游戏记录(应该很少)

## 评估指标
请查看final_test.log中的统计数据:
- 平均分数 (目标: >10,000)
- 最高分数
- 256/512/1024达成率
- 避免胜利成功率 (目标: >95%)
EOF

echo "训练报告已生成: $LOG_DIR/training_report.md"
echo
echo "查看最终测试结果:"
echo "tail -10 $LOG_DIR/final_test.log"
echo
echo "查看详细游戏记录:"
echo "tail -20 normal_games.log"