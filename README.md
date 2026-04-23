# RecBole Recommendation System Evaluation Framework

基于 [RecBole](https://github.com/RUCAIBox/RecBole) 构建的推荐系统算法研究评估框架，用于系统性地比较和复现推荐算法。

---

## TODO

> 当前进度：项目骨架已初始化，仓库已绑定 GitHub。以下为待完成事项，按优先级排列。

### 阶段一：项目骨架搭建

- [x] 初始化 git 仓库并绑定 GitHub（`chillanthony/rec-framework`）
- [x] 编写 `.gitignore`（排除 dataset/、results/、log/ 等大文件目录）
- [x] 编写 `README.md`（框架设计说明、测试策略、部署流程）
- [ ] 导出并提交 `environment.yml` / `requirements.txt`（固定依赖版本）
- [ ] 创建本地目录：`dataset/`、`results/`、`saved/`、`log/`（不纳入 git）

### 阶段二：配置文件

- [ ] 编写 `configs/base.yaml`（公共配置：seed、metrics、eval_args、负采样）
- [ ] 编写 `configs/debug.yaml`（本地测试专用：2 epoch、uni100 评估模式）
- [ ] 编写模型配置：`configs/models/BPR.yaml`、`LightGCN.yaml`、`SASRec.yaml` 等
- [ ] 编写数据集配置：`configs/datasets/ml-1m.yaml`、`amazon-beauty.yaml` 等

### 阶段三：核心脚本

- [ ] 编写 `scripts/run_single.py`（单模型单数据集运行，支持多 config 文件合并）
- [ ] 编写 `scripts/run_batch.py`（批量枚举模型 × 数据集，支持 `--dry-run`）
- [ ] 编写 `scripts/summarize_results.py`（解析 log/ 输出 CSV + LaTeX 表格）

### 阶段四：源码模块

- [ ] 创建 `src/` 目录结构（`custom_models/`、`custom_datasets/`、`utils.py`）
- [ ] 在 `src/utils.py` 中实现日志解析函数（从 RecBole log 提取指标数值）
- [ ] （按需）实现自定义模型，继承 RecBole 基类放入 `src/custom_models/`

### 阶段五：本地可用性测试

- [ ] 下载小数据集（如 `ml-100k`）用于本地测试
- [ ] 本地 CPU 跑通单个模型完整流程（加载→训练→验证→测试→写结果）
- [ ] 本地跑通 `run_batch.py --dry-run`，确认枚举逻辑正确
- [ ] 通过本地测试检查清单所有条目

### 阶段六：服务器部署与正式实验

- [ ] 在服务器上还原 conda 环境（`conda env create -f environment.yml`）
- [ ] 确定数据集上传方案（rsync 断点续传 / 服务器直接 wget）
- [ ] 上传数据集并验证完整性（`md5sum -c dataset/MANIFEST.md5`）
- [ ] 服务器跑通完整实验流程（tmux 后台运行）
- [ ] 验证 `summarize_results.py` 能正确汇总服务器产出的 log

---

## 目录

- [环境配置](#环境配置)
- [框架设计建议](#框架设计建议)
- [目录结构](#目录结构)
- [本地 CPU 可用性测试](#本地-cpu-可用性测试)
- [服务器实验部署](#服务器实验部署)
- [数据集上传策略](#数据集上传策略)

---

## 环境配置

```bash
conda activate sparserec-dev
pip install recbole
```

当前环境：`sparserec-dev`，RecBole 版本：`1.2.1`

---

## 框架设计建议

### 核心原则

1. **配置驱动**：所有实验参数通过 YAML 配置文件管理，避免硬编码。
2. **可复现性**：固定随机种子，记录完整超参数，使用 RecBole 的内置日志。
3. **模块化**：将数据处理、模型训练、评估、结果汇总解耦为独立模块。
4. **批量实验**：一条命令跑多个模型 × 多个数据集的组合，结果自动汇总为表格。

### 推荐框架结构

```
rec-framework/
├── configs/                  # 实验配置（每个模型/数据集一个 YAML）
│   ├── base.yaml             # 公共基础配置（seed, metrics, eval_args 等）
│   ├── models/
│   │   ├── BPR.yaml
│   │   ├── LightGCN.yaml
│   │   └── SASRec.yaml
│   └── datasets/
│       ├── ml-1m.yaml
│       └── amazon-beauty.yaml
├── dataset/                  # 数据集目录（不纳入 git，见下方上传策略）
├── scripts/
│   ├── run_single.py         # 运行单个实验
│   ├── run_batch.py          # 批量运行所有组合
│   └── summarize_results.py  # 汇总结果为 CSV/LaTeX 表格
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

### 配置文件设计

**`configs/base.yaml`（公共配置）**

```yaml
# 可复现性
seed: 2024
reproducibility: True

# 评估指标（推荐 Top-K 评估）
metrics: ['Recall', 'NDCG', 'Precision', 'Hit', 'MRR']
topk: [5, 10, 20]
valid_metric: NDCG@10

# 评估策略
eval_args:
  split: {'RS': [0.8, 0.1, 0.1]}   # 随机分割
  group_by: user
  order: RO                         # Random Order
  mode: full                        # full ranking

# 负采样（训练时）
train_neg_sample_args:
  distribution: uniform
  sample_num: 1
```

**`configs/models/LightGCN.yaml`（模型特定配置）**

```yaml
# 继承 base.yaml（在 run_single.py 中合并）
model: LightGCN
n_layers: 3
reg_weight: 1e-4
epochs: 100
learning_rate: 1e-3
train_batch_size: 4096
```

### 批量实验脚本设计思路

```python
# scripts/run_batch.py 核心逻辑示意
MODELS = ['BPR', 'LightGCN', 'SASRec', 'BERT4Rec']
DATASETS = ['ml-1m', 'amazon-beauty']

for model, dataset in itertools.product(MODELS, DATASETS):
    config = merge_configs('configs/base.yaml',
                           f'configs/models/{model}.yaml',
                           f'configs/datasets/{dataset}.yaml')
    run_recbole(model=model, dataset=dataset, config_dict=config)
    # 解析 log/ 下的结果，写入 results/summary.csv
```

### 关键设计决策

| 问题 | 建议方案 |
|------|----------|
| 多次运行取均值 | 在脚本中循环不同 seed，结果取 mean ± std |
| 超参搜索 | 用 Optuna 或手动网格搜索，在 YAML 中定义候选值 |
| 实验追踪 | 用 RecBole 内置日志 + `summarize_results.py` 解析；可选接入 W&B |
| 自定义模型 | 继承 `recbole.model.abstract_recommender` 基类，放入 `src/custom_models/` |
| 结果汇报 | `summarize_results.py` 输出 LaTeX 表格，直接粘入论文 |

---

## 目录结构

见 [推荐框架结构](#推荐框架结构)。

---

## 本地 CPU 可用性测试

在将代码推送到服务器跑完整实验前，**务必在本地 CPU 上验证整个流程**，避免在服务器上浪费时间排查环境问题。

### 测试策略：Mini 数据集 + 少量 Epoch

核心思路是**不改代码，只用配置控制规模**，确保本地测试与服务器实验走完全相同的代码路径。

**`configs/debug.yaml`（本地测试专用配置）**

```yaml
# 覆盖所有影响速度的参数
epochs: 2                    # 只跑 2 轮，验证训练/验证/测试流程完整性
train_batch_size: 64
eval_batch_size: 64
eval_args:
  split: {'RS': [0.8, 0.1, 0.1]}
  mode: uni100               # 用 100 个负样本替代 full ranking，大幅加速评估
```

**运行本地测试**

```bash
conda activate sparserec-dev

# 单个模型快速测试（用小数据集 ml-100k）
python scripts/run_single.py \
    --model LightGCN \
    --dataset ml-100k \
    --config_files configs/base.yaml configs/models/LightGCN.yaml configs/debug.yaml

# 验证批量脚本逻辑（dry-run 模式，只打印命令不执行）
python scripts/run_batch.py --dry-run
```

### 本地测试检查清单

- [ ] 数据集能正确加载（无格式错误）
- [ ] 前向传播无报错（模型结构正确）
- [ ] 训练循环完整跑完（loss 正常下降）
- [ ] 验证集评估正常输出指标
- [ ] 测试集最终评估结果能写入 `results/`
- [ ] 批量脚本能正确枚举所有模型×数据集组合
- [ ] 日志文件正确生成到 `log/`

### 注意事项

- 本地测试只验证**流程正确性**，不验证指标数值（CPU 上 2 epoch 的结果没有参考意义）
- 若模型含 GPU 专用操作（如自定义 CUDA kernel），在 CPU 上测试时需在配置中设置 `use_gpu: False`
- RecBole 会自动检测 GPU，CPU 环境下无需额外配置

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
# 方式一：通过 git（推荐，代码有版本控制）
git push origin main
# 服务器上
git clone <your-repo-url>
# 或
git pull origin main

# 方式二：rsync（适合频繁小改动，排除大文件）
rsync -avz --exclude='dataset/' --exclude='results/' --exclude='log/' \
    ./ user@server:/path/to/rec-framework/
```

### 服务器上运行

```bash
# 后台运行，输出重定向（防止 SSH 断开丢失进度）
nohup python scripts/run_batch.py > run.log 2>&1 &
echo $! > run.pid   # 保存进程 ID

# 或者用 tmux（推荐，可随时 attach 查看进度）
tmux new-session -d -s rec_exp "python scripts/run_batch.py"
tmux attach -t rec_exp

# 查看进度
tail -f run.log
# 或
tail -f log/<latest-log-file>.log
```

---

## 数据集上传策略

数据集通常是大文件，以下是最高效的上传方案：

### 方案一：rsync（推荐首选）

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

**优点**：支持断点续传（中断后重新执行只传差异部分），适合大量文件或反复同步。

### 方案二：scp（简单直接）

```bash
# 传单个压缩包
scp -C dataset.tar.gz user@server:/path/to/rec-framework/

# 服务器上解压
ssh user@server "cd /path/to/rec-framework && tar -xzf dataset.tar.gz"
```

### 方案三：先压缩再上传（最节省带宽）

```bash
# 本地压缩
tar -czf dataset.tar.gz dataset/

# 上传
rsync -avz --progress dataset.tar.gz user@server:/path/to/rec-framework/

# 服务器上解压
ssh user@server "cd /path/to/rec-framework && tar -xzf dataset.tar.gz && rm dataset.tar.gz"
```

### 方案四：使用学术数据集的官方源直接在服务器下载

对于 MovieLens、Amazon Review 等公开数据集，**直接在服务器上下载**比从本地上传更快：

```bash
# 在服务器上直接下载 RecBole 支持的数据集
# RecBole 提供了预处理好的数据集：https://recbole.io/dataset_list.html
wget https://drive.google.com/... -O dataset.zip   # 按实际链接
# 或使用 RecBole 的数据集下载工具
python -c "from recbole.utils import get_local_time; print('ok')"
```

### 选型建议

| 场景 | 推荐方案 |
|------|----------|
| 数据集 < 1GB | `scp` 直接传 |
| 数据集 > 1GB 或多次同步 | `rsync`，支持断点续传 |
| 网络不稳定 | `rsync`（中断重连后继续） |
| 公开学术数据集 | 直接在服务器上 `wget`/`curl` |
| 机构内网有共享存储 | 直接挂载或 `cp`，无需传输 |

### 数据集版本管理

大文件不纳入 git。在 `dataset/` 下维护一个轻量的清单文件：

```bash
# 生成数据集清单（纳入 git）
find dataset/ -type f | sort | xargs md5sum > dataset/MANIFEST.md5
git add dataset/MANIFEST.md5
```

这样团队成员可以通过 `md5sum -c dataset/MANIFEST.md5` 验证本地数据集完整性。

---

## Git 工作流

```bash
# 初始化（已完成）
git init
git checkout -b main

# 关联 GitHub 远程仓库
git remote add origin https://github.com/chillanthony/rec-framework.git

# 首次提交
git add .
git commit -m "init: project scaffold and README"
git push -u origin main
```

---

## 参考资料

- [RecBole 官方文档](https://recbole.io/docs/)
- [RecBole GitHub](https://github.com/RUCAIBox/RecBole)
- [RecBole 数据集列表](https://recbole.io/dataset_list.html)
- [RecBole 配置参数说明](https://recbole.io/docs/user_guide/config_settings.html)
