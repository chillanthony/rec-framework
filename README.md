# RecBole Recommendation System Evaluation Framework

---

## 环境配置

先按平台安装 PyTorch，再安装其余依赖：

```bash
conda create -n rec-env python=3.11 -y

# 本地（CPU-only，macOS/Linux）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# 服务器（CUDA 12.1）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
pip install -r requirements.txt
```

---

## 框架设计

### 核心原则

1. **配置驱动**：所有实验参数通过 YAML 配置文件管理，避免硬编码。
2. **可复现性**：固定随机种子，记录完整超参数，使用 RecBole 的内置日志。
3. **模块化**：将数据处理、模型训练、评估、结果汇总解耦为独立模块。
4. **批量实验**：一条命令跑多个模型 × 多个数据集的组合，结果自动汇总为表格。

### 推荐框架结构

> `run_single.py` 用法速查：
> ```bash
> # 正式实验
> python scripts/run_single.py --model SASRec --dataset amazon-videogames-2023-5c-llo
> # 本地冒烟测试（CPU、2 epoch、uni100 eval，设置内嵌于 tiny 数据集 yaml）
> python scripts/run_single.py --model SASRec --dataset amazon-videogames-2023-5c-llo-tiny
> # 覆盖任意参数（最高优先级）
> python scripts/run_single.py --model SASRec --dataset amazon-videogames-2023-5c-llo --params learning_rate=0.005
> ```
>
> `summarize_results.py` 用法速查：
> ```bash
> # 扫 log/ 下所有 .log，输出 HR@{10,20} + NDCG@{10,20} 对比表到 results/summary.md
> python scripts/summarize_results.py
> # 自定义日志根目录与输出路径
> python scripts/summarize_results.py --log_root log/ --out results/summary.md
> ```
> 解析失败的日志（如训练中断、未跑完 test）路径会写入 `results/parse_failures.txt`，不影响主流程。

> `make_tiny_dataset.py` 用法速查：
> ```bash
> # 默认：从 amazon-videogames-2023-5c-llo 中采样 200 个用户，写入 *-tiny
> python scripts/make_tiny_dataset.py
> # 自定义源数据集与采样规模
> python scripts/make_tiny_dataset.py --src amazon-scientific-2018 --n_users 100
> # 指定目标数据集名（默认 <src>-tiny）
> python scripts/make_tiny_dataset.py --src amazon-videogames-2023-5c-llo --dst my-tiny
> # 自定义随机种子（保证可复现）
> python scripts/make_tiny_dataset.py --seed 42
> ```

```
rec-framework/
├── configs/                  # 实验配置（每个模型/数据集一个 YAML）
│   ├── base.yaml             # 公共基础配置（seed, metrics, eval_args 等）
│   ├── models/
│   │   └── ....yaml
│   └── datasets/
│       ├── ....yaml
├── dataset/                  # 数据集目录（不纳入 git，见下方上传策略）
├── scripts/
│   ├── run_single.py         # 运行单个实验
│   ├── make_tiny_dataset.py  # 为任意数据集生成用于 debug 的微型子集
│   └── summarize_results.py  # 扫日志，汇总 HR/NDCG@{10,20} 对比表到 Markdown
├── src/
│   ├── custom_models/        # 自定义/改进的模型（继承 RecBole 基类）
│   ├── custom_datasets/      # 自定义数据预处理逻辑
│   └── utils.py              # 工具函数（日志解析、结果写入等）
├── results/                  # 实验结果（不纳入 git）
├── saved/                    # RecBole 保存的模型检查点（不纳入 git）
├── log/                      # RecBole 日志（不纳入 git）
├── requirements.txt
└── README.md
```

---

## 本地 CPU 可用性测试

在将代码推送到服务器跑完整实验前，**务必在本地 CPU 上验证整个流程**，避免在服务器上浪费时间排查环境问题。

核心思路是**不改代码，只用配置控制规模**，确保本地测试与服务器实验走完全相同的代码路径。

三层提速手段叠加，端到端目标 < 30 秒：

| 手段 | 加速来源 | 配置位置 |
|------|----------|----------|
| `epochs: 2` | 训练轮数减少 | `configs/datasets/amazon-videogames-2023-5c-llo-tiny.yaml` |
| `mode: uni100` | 评估用 100 个负样本替代 full ranking（约 100× 加速） | `configs/datasets/amazon-videogames-2023-5c-llo-tiny.yaml` |
| 微型数据集（~200 用户） | 数据加载 / tokenization 从数十秒缩短到 < 3 秒 | `dataset/amazon-videogames-2023-5c-llo-tiny/` |

```bash
# 从完整数据集中随机抽取 200 个用户，写入 dataset/amazon-videogames-2023-5c-llo-tiny/
python scripts/make_tiny_dataset.py

# 参数说明（均有默认值，通常无需修改）：
#   --src      源数据集名称（默认：amazon-videogames-2023-5c-llo）
#   --n_users  采样用户数（默认：200）
#   --seed     随机种子（默认：2024，保证可复现）
```

```bash
python scripts/run_single.py --model SASRec --dataset amazon-videogames-2023-5c-llo-tiny
```

---

## 服务器实验部署

### 环境同步

参考 "环境配置" 章节，确保服务器环境与本地一致。

### 代码同步

```bash
git push origin main
# 服务器上
git clone <your-repo-url>
# 或
git pull origin main
```

### 数据集上传

```bash
# 断点续传、只传增量、压缩传输
rsync -avz --progress \
    dataset/ \
    user@server:/path/to/rec-framework/dataset/

# 参数说明：
# -a  归档模式（保留权限、时间戳等）
# -v  显示详情
# -z  传输时压缩
# --progress  显示进度条
```

### tmux 实验管理

SSH 断线后实验继续运行的核心手段。所有服务器实验均应在 tmux session 内启动。


```bash
# 创建命名 session 并启动实验
tmux new -s sasrec
python scripts/run_single.py --model SASRec --dataset amazon-videogames-2023-5c-llo

# 脱离（实验继续跑，SSH 可安全断开）
Ctrl+b d

# 回来查看进度
tmux attach -t sasrec
```


| 操作 | 快捷键 / 命令 |
|------|--------------|
| 脱离 session | `Ctrl+b d` |
| 重连 session | `tmux attach -t <name>` |
| 列出所有 session | `tmux ls` |
| 水平分割面板 | `Ctrl+b %` |
| 垂直分割面板 | `Ctrl+b "` |
| 切换面板 | `Ctrl+b 方向键` |
| 关闭当前面板 | `exit` |
| 强制关闭 session | `tmux kill-session -t <name>` |

