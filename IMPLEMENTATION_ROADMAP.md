# MiniOPD Implementation Roadmap

> 项目名称：MiniOPD：面向小模型数学推理的低成本 On-Policy Distillation
>
> 本文档是项目实施的主参考文件。每个阶段按“目标 → 小步骤 → 产出物 → 验收条件”推进；上一阶段未验收时，不进入下一阶段。实验结果必须来自真实运行，禁止用目标值或公开结果替代实测结果。

## 0. 项目目标与固定边界

### 0.1 主训练路线

```text
Qwen2.5-0.5B-Instruct
    → 统一 Base 评测
    → 教师生成数学推理回答
    → Cold-start SFT
    → Sampled-token OPD
    → GSM8K 统一评测
    → 准确率、Overlap、Entropy Gap 与成本分析
```

### 0.2 固定模型与数据

- 学生：`Qwen/Qwen2.5-0.5B-Instruct`
- 教师：`Qwen/Qwen2.5-Math-1.5B-Instruct`
- 数据集：`openai/gsm8k`
- 第一阶段只完成一组模型、一套数据和一条主训练路线。

### 0.3 第一阶段不做

- 不更换教师或学生模型；
- 不使用 Qwen3 或 7B 教师；
- 不加入 GRPO/task reward；
- 不扩展多个数学数据集；
- 不做多随机种子；
- 不在主线中加入 Top-K loss；
- 不做 intersection/union 消融；
- 不追求长上下文和复杂原创算法。

### 0.4 当前算法决策

```yaml
opd:
  type: sampled-token
  loss_mode: k1
  use_policy_gradient: true
  use_task_rewards: false

rollout:
  n: 2
  temperature: 1.0
  top_p: 1.0
  max_response_length: 512

teacher:
  temperature: 1.0
  frozen: true
```

- `k1` 表示 sampled-token reverse-KL estimator。
- 训练 rollout 必须随机采样，不能使用 greedy/Top-1。
- 教师只在学生轨迹上评分，不生成自己的替代轨迹。
- Top-16 overlap 和 entropy 作为独立诊断，不进入主训练 loss。
- Top-4/Top-8/Top-16 和 task reward 只在主实验完成后作为扩展。

---

## 1. 理解参考项目与 OPD 原理

### 1.1 学习目标

- 理解 on-policy、off-policy 和 policy lag；
- 理解学生 rollout 与教师 prefix scoring；
- 理解 next-token logits shift；
- 理解 prompt、response、padding、EOS mask；
- 理解 old student log-prob、current student log-prob 和 teacher log-prob；
- 理解 sampled-token 与 Top-1 的区别；
- 理解 actor、rollout engine、teacher 和权重同步。

### 1.2 参考文件

- `Rethinking-OPD-main/README.md`
- `Rethinking-OPD-main/on_policy_distillation.sh`
- `Rethinking-OPD-main/verl_example/opd.sh`
- `Rethinking-OPD-main/verl/verl/trainer/ppo/ray_trainer.py`
- `Rethinking-OPD-main/verl/verl/workers/actor/dp_actor.py`
- `Rethinking-OPD-main/verl/verl/workers/fsdp_workers.py`
- `Rethinking-OPD-main/verl/verl/trainer/ppo/core_algos.py`

### 1.3 必须能解释的数据流

```text
GSM8K prompt
    → 统一 chat template
    → 当前学生生成 response
    → 保存 sampled token 和 old student log-prob
    → 教师读取完全相同的 prompt + student response
    → 教师计算 sampled token log-prob
    → reward = teacher_logp - old_student_logp
    → response_mask 排除 prompt/padding/EOS 后区域
    → reward 作为固定 advantage
    → current student log-prob 接收梯度
    → actor 更新
    → 新权重同步到 rollout engine
```

### 1.4 验收条件

- 能解释为什么 OPD 必须沿学生轨迹评分；
- 能手算一个小词表下的 sampled-token reward；
- 能解释 logits shift、stop-gradient 和权重同步；
- 能解释 sampled-token 为什么不是 Top-1。

---

## 2. 确定 verl 接入与硬件架构

### 2.1 verl 接入决策

- 使用上游 verl 官方 `distillation.*` 接口；
- 锁定经过 smoke test 的准确 Git commit；
- 不复制论文仓库的定制版 verl；
- MiniOPD 独立维护配置、指标、测试、评测与辅助代码；
- 在 README 中注明参考项目、改编内容和许可证。

### 2.2 学习官方 OPD 配置

- `distillation.enabled`
- `distillation.teacher_models.*`
- `distillation.distillation_loss.loss_mode=k1`
- `distillation.distillation_loss.use_policy_gradient=true`
- `distillation.distillation_loss.use_task_rewards=false`
- `loss_max_clamp`
- `log_prob_min_clamp`

### 2.3 正式双卡规划

```text
硬件：2 × NVIDIA RTX 4090 24GB
GPU 0：0.5B student actor + student rollout
GPU 1：1.5B frozen teacher inference
```

- 第一版不把 0.5B 学生跨两卡 FSDP 分片；
- 实际 placement、offload、vLLM 显存比例和 batch size 由 smoke test 决定；
- PyTorch、vLLM、verl 和 CUDA 运行时版本在实际实例启动后锁定。

### 2.4 产出物

- `reports/architecture_decision.md`
- verl commit 与依赖版本表；
- student/rollout/teacher 资源关系图；
- 初始显存预算与未知风险清单。

### 2.5 验收条件

- 能区分官方 distillation 接口与旧 `reward_model + token_reward_direct` 路线；
- 能解释 `k1` 为什么要通过 policy-gradient surrogate 更新；
- 能解释为什么 sampled-token 主线不自动提供 Top-16 诊断；
- 明确正式训练硬件和两张 GPU 的职责。

---

## 3. 建立独立、可安装的项目骨架

### 3.1 目标结构

```text
miniopd/
├── data/
├── miniopd/
│   ├── data/
│   ├── evaluation/
│   ├── generation/
│   ├── losses/
│   ├── metrics/
│   ├── trainers/
│   └── cli.py
├── configs/
├── scripts/
├── tests/
├── results/
├── reports/
├── pyproject.toml
├── uv.lock
├── Makefile
├── LICENSE
└── README.md
```

### 3.2 小步骤

1. 初始化 Git 仓库和 `.gitignore`；
2. 创建 Python package 和 CLI；
3. 配置 `pyproject.toml`、pytest、Ruff 和类型检查；
4. 将基础运行、GPU 训练和开发依赖分组；GPU 训练依赖在硬件环境确认后锁定；
5. 建立配置加载与日志约定；
6. 规定 raw/intermediate/final 数据目录；
7. 规定 checkpoint、results、reports 命名方式；
8. 添加基础 CI 或本地检查命令。

### 3.3 验收条件

- `uv sync` 或等价安装成功；
- `python -m miniopd --help` 成功；
- `pytest`、Ruff 基础检查成功；
- 在正式 GPU 服务器上完成依赖安装与基础检查。

---

## 4. GSM8K 数据准备与固定划分

### 4.1 锁定数据

- 下载 `openai/gsm8k`；
- 固定 dataset revision；
- 保存原始 schema、revision 和校验信息；
- 禁止手动覆盖原始数据。

### 4.2 固定划分

使用 `seed=42` 对 GSM8K train 7473 条划分：

- 3000：cold-start SFT prompts；
- 3473：OPD train prompts；
- 1000：validation；
- 官方 test 1319 条：最终测试。

### 4.3 Split manifest

每条记录至少包含：

- 原始 index；
- split；
- question hash；
- dataset revision；
- seed；
- 生成代码版本。

### 4.4 数据检查

- 各训练子集无重复；
- 标准化问题文本后再次查重；
- validation 不参与训练；
- test 不参与训练和调参；
- 检查空问题、异常答案和 schema。

### 4.5 格式转换

分别生成：

- 统一评测格式；
- teacher rollout 输入；
- SFT messages 格式；
- verl OPD parquet 格式。

### 4.6 测试与验收

- `tests/test_gsm8k_split.py`
- 行数、hash、seed 和无泄漏检查全部通过；
- 重复执行得到完全相同的 manifest。

---

## 5. 统一 Prompt 与答案解析

### 5.1 固定 Prompt

```text
System:
Please reason step by step, and put your final answer within \boxed{}.

User:
{question}
```

- 教师生成、学生 rollout、SFT、OPD 和评测必须使用同一语义模板；
- 保存 tokenizer 实际渲染后的 prompt 用于调试。

### 5.2 答案解析优先级

1. 最后一个 `\boxed{}`；
2. GSM8K `#### answer`；
3. 明确规则下的最后一个合法数值；
4. 无法解析标记为 `format_error`。

### 5.3 数值标准化

- 千分位逗号；
- 正负号；
- 整数、小数、分数；
- 百分号；
- 单位与尾部标点；
- 保留原始提取值和标准化值。

### 5.4 错误类型

- `wrong_answer`
- `format_error`
- `truncated`
- `empty_output`
- `repetition`
- `runtime_error`

### 5.5 测试与验收

- `tests/test_answer_parser.py`
- `tests/test_prompt_template.py`
- 人工边界案例全部通过；
- 无法解析的样本不会被静默忽略。

---

## 6. 建立统一评测管线

### 6.1 统一生成接口

- 支持 MLX、Transformers 和 vLLM 后端；
- 支持 greedy、batch generation 和断点续跑；
- 各后端输出同一种 JSONL schema。

### 6.2 正式评测配置

```yaml
temperature: 0
do_sample: false
max_new_tokens: 512
```

### 6.3 逐题记录

- question 与 ground truth；
- raw response；
- parsed answer；
- correctness；
- format status；
- truncated；
- input/output token 数；
- latency 与 tokens/s；
- 模型和配置标识。

### 6.4 汇总结果

- Exact Match；
- correct/total；
- format error；
- truncated ratio；
- 平均输出长度；
- tokens/s；
- 总耗时。

### 6.5 验收条件

- 相同 JSONL 重复评分结果一致；
- 后端差异不改变答案解析逻辑；
- 失败样本可以定位并人工复核。

---

## 7. GPU 小规模预评测

### 7.1 模型

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-Math-1.5B-Instruct`

### 7.2 100 题快速基准

- test 前 100 题；
- greedy；
- `max_new_tokens=512`；
- 使用正式训练环境的 BF16 推理配置；
- 使用统一 prompt 和答案解析器；
- 检查 accuracy、format error、truncated 和速度。

### 7.3 人工检查

- 随机正确样本；
- 随机错误样本；
- 所有 format error；
- 所有 truncated；
- prompt/chat template 是否符合预期。

### 7.4 小规模产出

- 保存逐题 JSONL 和汇总 JSON；
- 明确标注为 100 题预评测，不作为完整 test 正式结果。

### 7.5 验收条件

- 两个模型均完成可复查的 100 题结果；
- 没有未解释的批量 format error 或异常截断。

---

## 8. NVIDIA BF16 正式基准

### 8.1 环境准备

- 固定 CUDA、PyTorch、Transformers、vLLM、verl 版本；
- 检查 BF16、GPU 数量、驱动和磁盘空间；
- 记录 GPU 型号、数量与拓扑。

### 8.2 Preflight

- 模型 revision 存在；
- tokenizer/vocabulary 兼容；
- 数据 schema 正确；
- 输出目录可写且不会误覆盖；
- 单条生成成功；
- 小 batch 生成成功。

### 8.3 正式评测

- Base Student：完整 GSM8K test；
- Teacher：完整 GSM8K test；
- 同一 prompt、解析器和 greedy 配置。

### 8.4 验收条件

- 获得 Base 和 Teacher 的正式 BF16 JSONL/汇总结果；
- 不直接引用公开分数作为项目结果。

---

## 9. 教师生成 Cold-start SFT 数据

### 9.1 输入与生成配置

- 输入：3000 条 cold-start prompts；
- 教师：Qwen2.5-Math-1.5B-Instruct；
- `temperature=0.7`；
- `top_p=0.95`；
- `max_new_tokens=512`；
- 每题一条回答。

### 9.2 保存所有原始结果

- 不只保存成功样本；
- 每条保留 raw response、parsed answer、正确性、截断、格式和耗时；
- 生成过程支持断点续跑。

### 9.3 过滤规则

保留：

- 最终答案正确；
- 格式可解析；
- 未被长度截断；
- 非空；
- 无严重重复。

### 9.4 输出统计

- 输入数；
- 成功生成数；
- 正确数；
- 格式错误数；
- 截断数；
- 最终保留数；
- 生成耗时、tokens/s 和 GPU hours。

### 9.5 验收条件

- 人工抽查保留与丢弃样本；
- 过滤过程可复现；
- SFT 数据与 OPD/validation/test 无泄漏。

---

## 10. Cold-start SFT

### 10.1 初始训练方案

- 从 Qwen2.5-0.5B-Instruct 初始化；
- 优先全参数 BF16；
- epochs=1；
- learning rate=1e-5；
- effective batch size=32；
- gradient checkpointing=true；
- max length 根据真实 token 分布确定；
- 若显存不足才显式回退 LoRA。

### 10.2 SFT smoke test

- 32 条数据；
- 2～5 个 optimizer step；
- 验证 forward、backward、checkpoint、resume 和生成。

### 10.3 正式 SFT

- 记录 loss、gradient norm、学习率、吞吐和峰值显存；
- 保存完整配置、命令与 checkpoint；
- 使用 validation 选择 checkpoint。

### 10.4 SFT-only 评测

- 在完整 GSM8K test 上使用统一 BF16 管线；
- 保存 Base → SFT 的真实变化和失败案例。

### 10.5 验收条件

- SFT checkpoint 可独立加载；
- validation 和 test 评测完成；
- 没有使用 test 选择 checkpoint。

---

## 11. Sampled-token OPD 算法测试

### 11.1 必须实现的测试

- `tests/test_token_shift.py`
- `tests/test_response_mask.py`
- `tests/test_sampled_opd_reward.py`
- `tests/test_teacher_no_grad.py`
- `tests/test_overlap_metric.py`
- `tests/test_entropy_metric.py`
- `tests/test_weight_sync.py`

### 11.2 核心测试要求

1. 人工小词表手算 log-prob 和 reward；
2. 验证位置 `t` logits 预测位置 `t+1` token；
3. prompt、padding、EOS 后位置不进入 loss；
4. 教师参数无梯度且不变化；
5. reward 符号正确；
6. 极端 log-prob 不产生 NaN/Inf；
7. overlap 和 entropy 与人工计算一致；
8. actor 更新后 rollout 权重确实刷新。

### 11.3 验收条件

- 所有算法单元测试通过；
- 对每个关键 tensor 记录 shape、dtype、device 和 mask 语义。

---

## 12. 接入 verl Sampled-token OPD

### 12.1 初始化与数据

- 从选定的 SFT checkpoint 初始化 student；
- 使用 3473 条 OPD prompts；
- 不混入 SFT prompts、validation 或 test。

### 12.2 Student rollout

```yaml
n: 2
temperature: 1.0
top_p: 1.0
max_prompt_length: 256
max_response_length: 512
```

### 12.3 Teacher scoring

- 教师读取完全相同的 student trajectory；
- teacher temperature=1.0；
- no-grad；
- 只返回训练所需 sampled-token log-prob。

### 12.4 OPD loss

```yaml
loss_mode: k1
use_policy_gradient: true
use_task_rewards: false
learning_rate: 1.0e-6
gradient_clip: 1.0
epochs: 1
```

### 12.5 权重同步

- 每批 rollout 后尽快更新；
- actor 更新后同步到 rollout engine；
- 记录 actor/rollout model version；
- 避免未记录的 policy lag。

### 12.6 验收条件

- 单个 batch 完成 rollout → scoring → reward → backward → update → sync；
- reward 和 mask 可人工抽查；
- 教师与学生 tokenizer 对齐。

---

## 13. 训练诊断指标

### 13.1 每步基础指标

- token reward mean/std/min/max；
- policy/distillation loss；
- gradient norm；
- response length；
- rollout accuracy；
- truncated ratio；
- format error ratio；
- rollout、teacher scoring 和 training tokens/s；
- GPU memory。

### 13.2 周期性 Top-16 诊断

在固定小批量、相同 student prefixes 上计算：

- student Top-16；
- teacher Top-16；
- overlap ratio；
- overlap-token advantage；
- student/teacher overlap probability mass。

### 13.3 Entropy 诊断

- exact student entropy；
- exact teacher entropy；
- absolute entropy gap；
- 若使用近似 entropy，必须在指标名称中明确标注。

### 13.4 位置分段

按有效 response token 划分：

- 前 1/3；
- 中 1/3；
- 后 1/3。

分别记录 teacher entropy、student entropy、token reward 和 overlap，检查后段奖励退化。

### 13.5 验收条件

- 诊断不进入训练 loss；
- 诊断频率可配置；
- 指标使用相同 mask；
- 不因诊断造成不可接受的训练开销。

---

## 14. 端到端 OPD Smoke Test

### 14.1 Smoke 配置

- 32 条 train；
- 8 条 validation；
- max response length=128；
- rollout n=1；
- 2～5 个 optimizer step；
- 小 batch/micro batch。

### 14.2 必须验证

1. 数据加载；
2. student rollout；
3. teacher scoring；
4. sampled-token reward；
5. mask 与 shift；
6. backward；
7. optimizer step；
8. actor→rollout 权重同步；
9. checkpoint；
10. resume；
11. validation；
12. 日志与诊断；
13. 无 NaN/Inf。

### 14.3 验收条件

- 一条命令可重复完成 smoke test；
- 退出码为 0；
- checkpoint 可恢复；
- 指标与样本输出可审计。

---

## 15. 正式 Sampled-token OPD

### 15.1 正式训练

- 使用完整 3473 条 OPD prompts；
- 从 SFT checkpoint 初始化；
- 运行 1 epoch；
- 保存 step 日志、checkpoint 和诊断结果。

### 15.2 Checkpoint 选择

- 只使用 validation；
- 记录选择规则；
- 禁止查看 test 后反向调参。

### 15.3 最终测试

统一评测：

- Base Student；
- Cold-start SFT；
- SFT + Sampled-token OPD；
- Teacher。

### 15.4 结果计算

```text
SFT Gain = SFT - Base
OPD Additional Gain = OPD - SFT
Gap Recovery = (OPD - Base) / (Teacher - Base)
```

### 15.5 效率统计

- GPU hours；
- 峰值显存；
- rollout tokens/s；
- teacher scoring tokens/s；
- training tokens/s；
- checkpoint 大小；
- 学生与教师推理速度。

### 15.6 验收条件

- 所有数字来自真实保存的实验产物；
- 可从配置和 checkpoint 复现；
- 失败与中断实验不被隐藏。

---

## 16. Top-K 扩展实验（主线完成后）

### 16.1 先明确 Top-K 定义

- 官方 verl：teacher Top-K forward KL；
- 论文主实现：student Top-K reverse KL；
- 两者不能用同一个模糊名称描述。

### 16.2 实验顺序

1. Top-4；
2. Top-8（低成本曲线）或 Top-16（与论文更易对照）；
3. 与 sampled-token 使用相同数据、初始化和训练预算。

### 16.3 对比指标

- GSM8K EM；
- overlap/entropy；
- 稳定性和 gradient norm；
- GPU hours；
- 峰值显存；
- teacher scoring 吞吐；
- 每种方法的额外实现复杂度。

---

## 17. Task Reward 混合实验（可选扩展）

### 17.1 对照组

```text
纯 Sampled-token OPD
vs.
Sampled-token OPD + GSM8K outcome reward
```

### 17.2 要求

- 明确命名为 hybrid OPD，不称为纯 OPD；
- 记录 distillation loss 和 task loss 各自权重；
- 分析 `n=2` 时“两错/一对一错/两对”的比例；
- 不把扩展结果覆盖主实验。

---

## 18. 最终报告、README 与求职展示

### 18.1 主结果表

| 模型阶段 | GSM8K EM | 相对 Base | 相对上一阶段 |
|---|---:|---:|---:|
| Base Student | 实测 | — | — |
| Cold-start SFT | 实测 | 实测 | 实测 |
| SFT + OPD | 实测 | 实测 | 实测 |
| Teacher | 实测 | — | — |

### 18.2 机制结果

| 指标 | OPD 开始 | OPD 结束 | 变化 |
|---|---:|---:|---:|
| Top-16 overlap | 实测 | 实测 | 实测 |
| Entropy gap | 实测 | 实测 | 实测 |
| Token reward | 实测 | 实测 | 实测 |
| Gradient norm | 实测 | 实测 | 实测 |

### 18.3 图表

- validation/test accuracy；
- overlap 曲线；
- student/teacher entropy；
- entropy gap；
- token reward；
- gradient norm；
- response length；
- 前/中/后段指标；
- 吞吐与显存。

### 18.4 失败案例

- format error；
- truncated；
- 重复生成；
- reward 极端值；
- teacher/student 明显分歧；
- OPD 后退化的问题；
- 失败配置与原因。

### 18.5 README 内容

1. 项目背景和 STAR 叙事；
2. OPD 原理；
3. 系统架构图；
4. 模型与数据选择；
5. 环境安装；
6. GPU 小规模预评测；
7. NVIDIA BF16 基准；
8. 教师生成与 SFT；
9. 一键 smoke test；
10. 完整 OPD 命令；
11. 结果表与训练曲线；
12. 失败案例；
13. 已知限制；
14. 上游项目、许可证和引用。

### 18.6 求职材料

- 一条精炼简历描述；
- STAR 版项目介绍；
- 3 分钟讲解；
- 10 分钟技术深挖；
- 常见面试追问；
- 关键代码讲解路径。

---

## 19. 全流程通用规则

每个阶段都必须：

1. 先明确输入、输出和验收条件；
2. 先运行小规模测试，再运行完整任务；
3. 保存实际命令、配置、日志和环境信息；
4. 报告真实结果，不把目标值写成结果；
5. 记录尚未验证的假设；
6. 不省略错误、中断和失败实验；
7. 不覆盖已有用户文件和实验产物；
8. 不使用 test 调参；
9. 不静默改变模型、算法或训练配置；
10. 重要改动必须更新本文档或对应 ADR。

## 20. 当前进度

- [x] 阅读参考项目与论文核心结论
- [x] 理解 sampled-token、Top-1、logits shift、mask 和权重同步
- [x] 初步确定 sampled-token 主线
- [ ] 锁定 verl commit 与正式硬件配置
- [ ] 建立完整项目骨架
- [ ] 实现 GSM8K 数据与统一评测
- [ ] 完成 GPU 小规模预评测
- [ ] 完成 NVIDIA BF16 基准
- [ ] 完成教师生成和 Cold-start SFT
- [ ] 完成 sampled-token OPD 单元测试与接入
- [ ] 完成端到端 smoke test
- [ ] 完成正式训练、评测与报告
- [ ] 完成可选 Top-K/Hybrid 扩展
