# Tag RL 实验汇总（自动迭代版）

本文档汇总当前项目中围绕 PPO 双角色（tagger / runner）的主要实验设计、参数迭代和结果结论，重点关注 **2B 角色隔离评估** 下“训练后是否优于 random”。

## 1. 实验目标

- 通过多轮消融确定对学习最有效的参数组合。
- 同时关注 tagger 与 runner 两个角色的学习质量。
- 在 `run_experiment.py` 的 **Phase 2b（role isolated）** 中验证：
  - tagger：`trained tagger vs random runners`，希望指标高于 random 基线。
  - runner：`trained runners vs random tagger`，希望指标低于 random 基线。

## 2. 评估标准

按每个 seed 分别判定：

- `tagger_pass`: `tagger_mean_tags_trained > tagger_mean_tags_random`
- `runner_pass`: `runner_mean_tags_trained < runner_mean_tags_random`
- `both_pass`: 上述两个条件同时满足

random 基线（`num_episodes=10`, `steps=700`）：

- seed 11: 0.5
- seed 22: 0.6
- seed 33: 0.8
- seed 44: 1.0
- seed 55: 0.3

说明：这里的数值单位均为 role 评估中的 `mean_tags`。

## 3. 主要实验分区与结果

### 3.1 参数粗搜索与细化（Round 3-7）

目录：

- `experiments/manual_round3_tagradius`
- `experiments/manual_round4_cooldown`
- `experiments/manual_round5_reward_grid`
- `experiments/manual_round6_refine`
- `experiments/manual_round7_coef_refine`

核心发现：

- `TAG_RADIUS` 在当前任务里是关键杠杆，`32` 明显优于更小半径（如 28）。
- `TAG_COOLDOWN_FRAMES` 在短预算下（0/5/10）影响不稳定，未形成强单调结论。
- 奖励参数在 `PROXIMITY_RADIUS ≈ 140`、`PROXIMITY_COEF ≈ 0.27` 附近表现最好。
- Round6 / Round7 对 Round5 最优邻域做了细化验证，`(140, 0.27)` 的均值更高且波动更低。

### 3.2 长预算验证（Round 8）

目录：

- `experiments/manual_round8_long_best`

代表参数：

- `TAG_RADIUS=32`
- `TAG_COOLDOWN_FRAMES=5`
- `PROXIMITY_RADIUS=140`
- `PROXIMITY_COEF=0.27`
- 训练预算：`epochs 10 20`, `steps_per_round 800`

2B 结果（5 seeds）：

- `tagger_pass = 5/5`
- `runner_pass = 2/5`
- `both_pass = 2/5`

结论：tagger 学习稳定，runner 仍是主要瓶颈。

### 3.3 runner 导向修正（Round 9-11）

目录：

- `experiments/manual_round9_runner_focus`
- `experiments/manual_round10_runner_recover/C`
- `experiments/manual_round10_runner_recover/D`
- `experiments/manual_round11_reward_balance`

主要策略：

- 增大 cooldown、减小 tag 半径、降低/调整 proximity 强度。
- 额外平衡奖励：提高 `SURVIVAL_BONUS`、提高 `TAG_PENALTY`、降低 `TAG_REWARD`。

关键结果：

- Round9：`both_pass = 1/5`（比 Round8 更差）
- Round10-C：`both_pass = 1/5`
- Round10-D：`both_pass = 2/5`（与 Round8 持平）
- Round11：`both_pass = 2/5`（仍未突破）

结论：runner 提升存在，但跨 seed 稳定性不足，整体未超过 `2/5` 的双角色同时达标率。

### 3.4 失败 seed 定向修复（Round 12）

目录：

- `experiments/manual_round12_targeted`

设计：

- 针对先前失败 seed（11/44/55）做更长训练（`epochs 20 40`）与更保守 runner 友好参数。

结果（3 seeds）：

- `tagger_pass = 3/3`
- `runner_pass = 1/3`
- `both_pass = 1/3`

说明：seed11 被修复，seed44/55 仍失败。

## 4. 当前阶段结论

- **已确认**：tagger 在当前 PPO 训练框架下普遍可以学好（高稳定性）。
- **未解决**：runner 的稳定泛化不足，导致 `both_pass` 长期卡在 `2/5` 左右。
- **经验上较优的基础参数**（用于后续继续迭代）：
  - `TAG_RADIUS=32`
  - `TAG_COOLDOWN_FRAMES=5~10`
  - `PROXIMITY_RADIUS=140`
  - `PROXIMITY_COEF=0.27`（runner 导向可下探到 `0.24~0.25` 再测）

## 5. 后续建议（针对“both_pass”优化）

- 在固定 `TAG_RADIUS=30~32` 下，重点搜索：
  - `SURVIVAL_BONUS`
  - `TAG_PENALTY`
  - `PROXIMITY_COEF`
- 训练预算建议保持 `epochs 20 40` 以上，以减少早期偶然最优轮次对 `best` checkpoint 的偏置。
- 报告主指标建议统一为：
  - `both_pass / N_seeds`
  - `runner_pass / N_seeds`
  - 每个 seed 的 `tag_best` 与 `runner_best`（便于定位失败 seed）。

---

如果后续继续自动迭代，建议在此文档末尾按 `Round 13+` 追加，保持可追溯实验日志。
