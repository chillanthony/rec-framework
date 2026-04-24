# RecBole Recommendation System Evaluation Framework


## TODO


- [ ] 在服务器上还原 conda 环境（`conda env create -f environment.yml`）
- [ ] 上传数据集并验证完整性
- [ ] 服务器跑通完整实验流程（tmux 后台运行）

- [ ] 编写 `scripts/summarize_results.py`（解析 log/ 输出 CSV + LaTeX 表格）
- [ ] 在 `src/utils.py` 中实现日志解析函数（从 RecBole log 提取指标数值）
- [ ] （按需）实现自定义模型，继承 RecBole 基类放入 `src/custom_models/`

---

## 环境配置

```bash
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

```
rec-framework/
├── configs/                  # 实验配置（每个模型/数据集一个 YAML）
│   ├── base.yaml             # 公共基础配置（seed, metrics, eval_args 等）
│   ├── models/
│   │   ├── BPR.yaml
│   │   ├── LightGCN.yaml
│   │   └── SASRec.yaml
│   └── datasets/
│       ├── amazon-videogames-2023-5c-llo.yaml
│       └── amazon-videogames-2023-5c-llo-tiny.yaml  # 本地冒烟测试专用（内嵌 debug 设置），由 make_tiny_dataset.py 生成
├── dataset/                  # 数据集目录（不纳入 git，见下方上传策略）
├── scripts/
│   ├── run_single.py         # 运行单个实验
│   └── make_tiny_dataset.py  # 为任意数据集生成用于 debug 的微型子集
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


### 关键设计思路

| 问题 | 建议方案 |
|------|----------|
| 多次运行取均值 | 在脚本中循环不同 seed，结果取 mean ± std |
| 超参搜索 | 用 Optuna 或手动网格搜索，在 YAML 中定义候选值 |
| 实验追踪 | 用 RecBole 内置日志 + `summarize_results.py` 解析；可选接入 W&B |
| 自定义模型 | 继承 `recbole.model.abstract_recommender` 基类，放入 `src/custom_models/` |
| 结果汇报 | `summarize_results.py` 输出 LaTeX 表格，直接粘入论文 |

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

-  检查清单

- [ ] 数据集能正确加载（无格式错误）
- [ ] 前向传播无报错（模型结构正确）
- [ ] 训练循环完整跑完（loss 正常下降）
- [ ] 验证集评估正常输出指标
- [ ] 测试集最终评估结果打印（`best valid` / `test result`）
- [ ] 日志文件正确生成到 `log/`

---

## 服务器实验部署

### 环境同步

```bash
# 本地导出环境
conda env export --no-builds | grep -v "^prefix" > environment.yml
# 或者只导出 pip 依赖
pip freeze > requirements.txt

# 服务器上还原
conda env create -f environment.yml
# 或
pip install -r requirements.txt
```

### 代码同步

```bash
git push origin main
# 服务器上
git clone <your-repo-url>
# 或
git pull origin main

### 服务器上运行

```bash

tmux new-session -d -s rec_exp "python scripts/run_batch.py"
tmux attach -t rec_exp

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