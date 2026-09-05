# lamet-agent

<p align="right">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <img src="docs/lamet-agent-demo.gif" alt="lamet-agent 演示" width="800" />
</p>

`lamet-agent` 是一个 Python-first 框架，用于构建可复现的**大动量有效理论**（**La**rge **M**omentum **E**ffective **T**heory，LaMET）和格点 QCD 分析工作流。

## 快速开始

本机需要已登录的 Codex CLI。Codex 不使用
`--api-key-file`。

```bash
git clone https://github.com/Greyyy-HJC/lamet-agent.git && cd lamet-agent
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip && python3 -m pip install -e ".[codex]"
```

请使用夸克网盘下载原始数据，链接：https://pan.quark.cn/s/e72c425c0f7f；提取码：Ua6r。

```bash
unzip data_lamet_agent.zip && \
cd data_lamet_agent && unzip data_pion_pdf_cg.zip
lamet-agent run examples/pion_pdf_cg_manifest.json \
  --provider codex --model gpt-5.6-luna
```

如需将本次运行结果与参考结果进行比较：

```bash
python plot_pion_pdf_compare.py
```

也可以使用 `uv` 创建环境并安装：

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[codex]"
```

### 其他提供商

API 提供商（`openai`、`anthropic`、`gemini`、`grok`、`deepseek`，或自定义的兼容 OpenAI 的 HTTP(S) URL）需要执行 `python3 -m pip install -e .`（不使用 `[codex]` extra），并通过 `--api-key-file` 或相应提供商的环境变量提供密钥。详情参见[提供商和模型](#提供商和模型)。

```bash
lamet-agent run examples/pion_pdf_cg_manifest.json \
  --provider openai --model gpt-5.6-luna \
  --api-key-file api.key
```

### Manifest 示例

| Manifest                                     | 工作流                                                         | 数据参考 |
| -------------------------------------------- | -------------------------------------------------------------- | -------- |
| `examples/pion_pdf_cg_manifest.json`         | Coulomb-gauge pion PDF，使用最小二乘关联函数分析。              | [^1]     |
| `examples/pion_pdf_cg_lanczos_manifest.json` | Coulomb-gauge pion PDF，使用 nested-bootstrap Lanczos 分析。    | [^1]     |
| `examples/pion_pdf_gi_manifest.json`         | Gauge-invariant pion PDF。                                     | [^1]     |
| `examples/pion_da_gi_manifest.json`          | Gauge-invariant pion DA，包含系统学变体。                       | [^2]     |
| `examples/kaon_da_gi_manifest.json`          | Gauge-invariant kaon DA，包含系统学变体。                       | [^2]     |

同一主机还提供其他示例压缩包；请选择与该 manifest 所使用的 `data_*` 目录相匹配的 zip 文件（`data_pion_pdf_cg`、`data_pion_pdf_gi`、`data_pion_da_gi`、`data_kaon_da_gi`）。

[^1]: Xiang Gao, Wei-Yang Liu, and Yong Zhao, [*Parton Distributions from Boosted Fields in the Coulomb Gauge*](https://arxiv.org/pdf/2306.14960), arXiv:2306.14960.
[^2]: Jun Hua et al., [*Pion and Kaon Distribution Amplitudes from Lattice QCD*](https://arxiv.org/pdf/2201.09173), arXiv:2201.09173.

## 命令行

`lamet-agent` 和 `python -m lamet_agent` 提供相同的接口：

```text
lamet-agent {validate,plan,run} ...
```

### 验证

```bash
lamet-agent validate MANIFEST
```

Validate 读取 JSON 或 JSONC，并且不与 LLM 通信。它检查 manifest envelope、stage contract、systematics 声明、job DAG、路径、关联函数 descriptor 和 kernel 参数。

### 规划

```bash
lamet-agent plan MANIFEST \
  --provider PROVIDER \
  [--model MODEL] \
  [--api-key-file FILE] \
  [--output FILE | --in-place]
```

Plan 通过与 LLM 进行交互式对话来补全不完整的 manifest。它会验证提议的修改，并给出最终的自然语言摘要，供用户明确确认。

选项：

- `--provider`：已注册的提供商名称，或兼容 OpenAI 的 HTTP(S) URL；
- `--model`：覆盖提供商的默认模型；
- `--api-key-file`：只包含 API key 的文本文件；
- `--output`：输出路径，默认为 `<manifest>.planned.json`；
- `--in-place`：在用户明确接受后覆盖源文件。

`--output` 与 `--in-place` 互斥。规划后的输出必须与源 manifest 位于同一目录，以保持相对输入路径的含义不变。

终端控制命令包括 `/show`、`/issues`、`/undo`、`/edit`、`/save`、`/help` 和 `/quit`。`Enter` 提交，`Shift+Enter` 插入换行，`Ctrl+C` 取消。

独立运行 Plan 时，会写入已接受的 manifest，然后退出，不会执行分析 stages。

### 运行

```bash
lamet-agent run MANIFEST \
  --provider PROVIDER \
  [--model MODEL] \
  [--api-key-file FILE] \
  [--progress {auto,stage,job,none}]
```

Run 在执行数值 stages 前先进行验证。如果验证失败，它会使用所选提供商进入 Plan。用户接受修复后的有效 manifest 后，数值执行将自动继续。

默认进度模式为 `auto`：

- `auto`：声明了 systematics 时显示 stage-level job progress；否则进度由各数值 job 自行管理；
- `stage`：每个 stage 显示一个 job 计数器；
- `job`：显示特定 stage 的数值进度；
- `none`：禁用进度条。

### 提供商和模型

#### Codex CLI

`codex` 提供商使用可选的 `openai-codex` 包以及当前机器上缓存的 Codex 登录状态。它不使用 API key。`--model` 是可选项，用于覆盖 Codex SDK 的默认模型。

#### 兼容 OpenAI 的 API

已注册的 API 提供商包括 `openai`、`anthropic`、`gemini`、`grok` 和 `deepseek`。每个提供商都从 `--api-key-file` 或相应环境变量中读取 API key：

| 提供商    | 环境变量            |
| --------- | ------------------- |
| OpenAI    | `OPENAI_API_KEY`    |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini    | `GEMINI_API_KEY`    |
| Grok      | `GROK_API_KEY`      |
| DeepSeek  | `DEEPSEEK_API_KEY`  |

已注册的 API 提供商都有默认模型；`--model` 可以覆盖该默认值。也可以直接把兼容 OpenAI 的 HTTP(S) base URL 作为提供商传入。非本地自定义 URL 必须指定 `--model`；如果本地 URL 的 `/models` endpoint 恰好返回一个 model id，则可以省略该参数。

## 核心思想

manifest 包含运行元数据以及按顺序排列的 stage job 列表映射。Job id 构成一个 DAG：关联函数 jobs 选择原始记录，后续 jobs 通过 `target`、`denominator`、`input` 和 `quasi` 等按角色命名的 inputs 使用先前输出。

预期的 agent 行为：

- 在执行数值计算前验证完整的已编写工作流。
- 参数确定后，以确定性方式运行数值 stages。
- 仅在工作流需要 fit 或区间建议时咨询 LLM。
- 写出中间 NetCDF 数据、诊断信息、图和 stage reports，使完整分析路径可以检查。
- 基于数值证据、一致性检查和选定文献完成最终 Review。

已实现的 stage families 通常按以下顺序编写：

1. `correlator_analysis`
2. `renormalization`
3. `fourier_transform`
4. `perturbative_matching`
5. `extrapolation`
6. `review`

部分工作流可以省略不需要的 stages。`stages` 下各键的顺序就是执行顺序；不存在单独的 `metadata.stages` 列表。

架构、文件所有权和贡献者工作流记录在 [`DEVELOPMENT.md`](DEVELOPMENT.md) 中。

## 中间数据（NetCDF）

stage 之间传递的数值 artifacts 存储为 NetCDF 文件。每个 array 都具有：

- 最前面的 `resample` 维度；
- 一种采样模式：`raw`、`jackknife`、`bootstrap` 或 `gvar`；
- `t`、`tsep`、`tau`、`z`、`x`、`a` 或 momentum 等物理维度和坐标；
- 以 attributes 形式存储的 ensemble 和 stage provenance。

Job 也可以使用外部 `{ "file": ".../output.nc" }` artifact 作为输入。

典型的每个 job 文件包括：

| 文件                  | 用途                                             |
| --------------------- | ------------------------------------------------ |
| `output.nc`           | 携带 samples 的主要数值结果。                    |
| `summary.json`        | 决策、诊断信息和已声明的 artifacts。             |
| `llm_transcript.md`   | 适用时记录 LLM requests 和 responses。           |
| `diagnostics/*`       | 候选表和数值诊断。                               |
| `plots/*`             | PDF/SVG 结果和 fit-quality 图。                  |

stage 目录还会获得汇总的 `report.md`；job 目录不写 report 文件。Review 会写出最终的 `review.md`、`review_bundle.json` 以及一致性/文献证据。

### 不依赖 lamet-agent 检查或读取数据

NetCDF 是自描述格式，可以使用 `ncdump`、Panoply 或 xarray 检查：

```python
import xarray as xr

array = xr.load_dataarray("output.nc", auto_complex=True)
print(array.dims)
print(array.coords)
print(array.attrs)
```

第一维始终是 `resample`；其余维度描述相应 stage report 中记录的物理布局。

## Manifest 示例

loader 接受 JSON 和 JSONC 注释。当前 manifest envelope 为：

```json
{
  "metadata": {
    "run_id": "pion_pdf_cg",
    "root_directory": "..",
    "artifacts_directory": "runs/pion_pdf_cg/artifacts",
    "random_seed": 1984,
    "workers": 4,
    "target_observable": "pdf",
    "parton": "quark",
    "resample_mode": "jackknife",
    "bin_size": 1,
    "sample_error_mode": "covariance"
  },
  "stages": {
    "correlator_analysis": {
      "defaults": {},
      "jobs": [
        {
          "id": "ca_p5",
          "inputs": {
            "correlators": [
              {"json": "examples/pion_pdf_cg_correlators.json", "id": "p5_2pt"},
              {"json": "examples/pion_pdf_cg_correlators.json", "id": "p5_3pt"}
            ]
          }
        }
      ]
    }
  },
  "systematics": {}
}
```

三个顶层 objects 分别是：

- `metadata`：全局运行路径、target identity、重采样、误差、seed 和 worker 数量；
- `stages`：有序的 stage/job graph；
- `systematics`：可选的、由 stage 所有的变体声明。

每个 stage 包含共享的 `defaults` 和有序的 `jobs` 列表。一个 job 包含全局 `id`、按角色命名的 `inputs`，以及直接写在 job 上的参数覆盖。Stage defaults 填充 job 中省略的字段；显式 job 值仍具有最高优先级。

输入值可以是：

- 先前 job 的 id；
- `{ "file": "path/to/output.nc" }`；
- `{ "json": "descriptor.json", "id": "correlator_record" }`；
- receiving contract 允许的数值常量；
- receiving role 允许多个来源时使用的有序列表。

对于使用 descriptor JSON record 的关联函数输入，请参阅[标准关联函数 HDF5 格式](#标准关联函数-hdf5-格式)中的输入文件约定。

未知字段、无效选项、损坏的 input roles、重复 id、向前 job 引用、缺失路径以及跨参数不一致都会被拒绝。

全局运行 metadata 字段包括：

- 必填：`run_id`、`root_directory`、`artifacts_directory`、`random_seed`、`workers`、`target_observable`、`resample_mode`、`sample_error_mode` 和 `bin_size`；
- `parton`，当前为 `quark`，默认值也是该值；
- `samples`，仅在 bootstrap 模式下必填；
- `parameter_recommendation_retries`，默认为每个 job 额外尝试一次。

`target_observable` 接受 `pdf`、`da` 和 `gpd`。`sample_error_mode` 接受 `covariance`、`variance` 以及仅用于 bootstrap 的 `one_sigma`。

目前 Fourier、matching 和 extrapolation 支持系统学变体。这些变体会在执行前展开为具体 jobs，并保存到 `resolved_manifest.json`。

## 标准关联函数 HDF5 格式

关联函数 jobs 从 descriptor JSON 中选择一条记录：

```json
{"json": "examples/pion_pdf_cg_correlators.json", "id": "p5_3pt"}
```

### Descriptor 示例

下面这个完整示例定义了一个两点关联函数：

```json
{
  "correlators": [
    {
      "id": "p5_2pt",
      "ensemble": {"series": "HISQ", "id": "HISQa060_X", "a_s": 0.06,
                   "a_t": 0.06, "L_s": 48, "L_t": 64, "m_pi": 0.3},
      "count": 109,
      "format": "hdf5",
      "path": "correlators/pion_2pt.h5",
      "dataset": "g5/g5/PX5PY0PZ0",
      "dataset_dims": ["t", "configuration"],
      "dims": ["configuration", "t"],
      "coords": {"t": [0, 1, 2, 3]},
      "selectors": {"source_operator": "g5", "sink_operator": "g5",
                    "momentum": "PX5PY0PZ0", "gfix": "CG"},
      "correlator_type": "two_point",
      "hadron": {"name": "pion"},
      "source_momentum": [5, 0, 0],
      "sink_momentum": [5, 0, 0],
      "current": null,
      "source_sink_separation": null
    }
  ]
}
```

`path` 给出相对于 descriptor JSON 的 HDF5 文件位置；`dataset` 给出其中 array 的位置：

```text
pion_2pt.h5
└── g5
    └── g5
        └── PX5PY0PZ0     dataset, shape (4, 109)
```

叶节点 axes 为 `(t, configuration)`，与 `dataset_dims` 的声明一致。

### Dataset 路径和维度

`dims` 以 `configuration` 开头，而 `coords` 提供其他所有 axes。`dataset_dims` 给出每个 HDF5 叶节点中存储的 axis 顺序。`dataset` template 可以将 coordinate values 放入 HDF5 路径：

```json
{
  "dataset": "g5/g5/gT_nonlocal/PX5PY0PZ0/tsep{tsep}/bT0/bz{z}",
  "dataset_dims": ["tau", "configuration"],
  "dims": ["configuration", "tsep", "tau", "z"],
  "coords": {"tsep": [8, 10, 12], "tau": [0, 1, 2, 3], "z": [0, 1, 2]}
}
```

它会将 `tsep` 和 `z` 展开到不同的叶节点：

```text
pion_3pt.h5
└── g5/g5/gT_nonlocal/PX5PY0PZ0
    ├── tsep8/bT0/{bz0,bz1,bz2}
    ├── tsep10/bT0/{bz0,bz1,bz2}
    └── tsep12/bT0/{bz0,bz1,bz2}
```

每个 `bz*` 叶节点存储 `(tau, configuration)`。组装后的输出顺序为 `(configuration, tsep, tau, z)`。group 名称本身不受限制；只有 `dataset` template 定义层级结构。

### 关联函数类型

| 类型          | 常见维度                      | 附加要求                                                                                  |
| ------------- | ----------------------------- | ----------------------------------------------------------------------------------------- |
| `two_point`   | `configuration, t`            | `current` 必须为 `null`。                                                                 |
| `three_point` | `configuration, tsep, tau, z` | 必须提供 `current`；应提供 `tsep` 维度或设置 `source_sink_separation`。                    |
| `qda`         | `configuration, t, z`         | 必须提供 `current`。                                                                      |

动量是整数三元组。非 null 的 `current` 恰好包含 `kernel_operator`、`parton` 和 `renormalization_scheme`。descriptor 是所有 coordinates 和 provenance fields 的权威来源。同一 job 中选定的 records 必须具有相同的 ensemble 和 configuration count。

完整 descriptors 参见 `examples/pion_pdf_cg_correlators.json` 和 `examples/pion_da_gi_correlators.json`。

## 跨 Stage 的 Manifest 语义

本节记录跨越 stage 边界的约定。各参数的具体细节仍由 stage contracts 和示例 manifests 定义。

### Ratio 重整化

Renormalization 将物理 `scheme` 与实现 `strategy` 分开：

- `scheme`：`ratio`、`hybrid` 或 `msbar`；
- `strategy`：`external_denominator` 或 `self_renormalization`。

external-denominator ratio 对保留的每个 target sample 逐点相除，

$$
h_s^R(z)=\frac{h_s^{\mathrm{target}}(z)}
{h_s^{\mathrm{denominator}}(z)}.
$$

denominator 可以是先前的 job、NetCDF 文件，或者 contract 允许的有限非零常量。Hybrid jobs 还使用 `zs_fm`、`m0_gev` 和 `delta_m_gev` 来衔接短程与长程 prescription。

Perturbative matching 有自己的 `scheme`，它必须与 `kernel_id` 中编码的 scheme token 一致。为 matching kernel 提供的参数会根据该 kernel 的要求进行验证。

### `inputs.correlators[].polarization` 和 Fourier sectors

关联函数 descriptors 保留 hadron、current、gauge-fixing/link convention、source 和 sink momentum 以及 polarization provenance。Renormalization 将这些 provenance 写入其 NetCDF 输出。Fourier 根据上游结果和 manifest metadata 推导 tail family 和 projection。

PDF jobs 使用 `unpolarized`、`helicity` 或 `transversity` polarization，以及 `valence`、`singlet` 或 `full` Fourier sector。DA 使用完整复数结果，并可以选择有序的 light/heavy endpoint flavor classes。GPD 通过可选的 `hermitian_partner` role 和 phase-transfer convention 支持成对的 forward/exchanged flows。

Fourier 输入的 `z` coordinates 是以 fm 为单位的物理距离。Tail ranges 使用 `zmin_fm`、`zmax_fm` 和 `zmax_ext_fm`；momentum 和 lattice provenance 从上游数据读取，而不作为编写的 stage 参数重复声明。

### 每个 job 的 hybrid `zs_fm`

hybrid switch distance 属于使用它的数值 job。应分别在 renormalization 和 perturbative matching 下声明。两者均可获得时，Review 会检查完整 manifest chain 中的值。外部 partial workflow 可能无法提供足够 provenance 来验证这种关系，此时会如实报告。

### `metadata.random_seed`、`metadata.samples`、`metadata.sample_error_mode`、`metadata.bin_size`、`metadata.workers`

这些字段是全局运行 sampling 和 parallelism 配置的来源：

- `random_seed` 为可复现的 bootstrap/jackknife 和运行时数值建议设置 seed；
- `samples` 选择 bootstrap replica 数量；
- `sample_error_mode` 选择 covariance、variance-only 或 median one-sigma summaries；
- `bin_size` 在重采样前对原始 configurations 求平均；
- `workers` 限制独立进程工作量。

## Self-Renormalization 策略

`strategy: "self_renormalization"` 将 reusable factor 的拟合与其在一个或多个 targets 上的应用分开。

fit job 使用 `inputs.reference`，它可以是不同 lattice spacings 的有序列表。该 job 使用选定的 coordinate-space MSbar kernel、QCD scale、finite correction 和 covariance regularization，在正 physical-z grid 上确定一个携带 samples 的 factor。前三个正 z 坐标定义当前实现的短程拟合范围。

apply job 使用 `inputs.target` 和 `inputs.zR`。当 factor 具有 `a` 维度时，它选择匹配的 lattice spacing，验证 scale 和 provenance，可选地在 `z=0` 处归一化，并逐 sample 应用选定的 ratio、hybrid 或 MSbar prescription。

覆盖范围由 `z_coverage_policy` 控制：

- `strict`：要求 target grid 完全位于拟合 factor 内；
- `intersection`：仅保留共同 grid；
- `extrapolate`：仅允许使用已实现的 quadratic finite-term tail 向更大的长程 z 补全。

### 工作流

```text
reference source(s)
        │
        ▼
┌──────────────────────────┐
│ self-renormalization fit │
│ type = fit               │
└────────────┬─────────────┘
             │ zR job id / output.nc
             ▼
┌──────────────────────────┐
│ target application       │
│ type = apply             │
└────────────┬─────────────┘
             ▼
 renormalized matrix element
```

### Manifest 结构

```json
{
  "stages": {
    "renormalization": {
      "defaults": {
        "strategy": "self_renormalization",
        "scheme": "ratio",
        "normalization": false,
        "kernel_id": "z_msbar_pdf_nlo",
        "kernel_parameters": {},
        "mu": 2.0,
        "LambdaQCD_gev": 0.1,
        "z_coverage_policy": "extrapolate"
      },
      "jobs": [
        {
          "id": "rn_factor",
          "type": "fit",
          "d": -0.08183,
          "inputs": {"reference": [{"file": "reference_a06.nc"}]}
        },
        {
          "id": "rn_target",
          "type": "apply",
          "d": 0.19,
          "m0_gev": -0.094,
          "inputs": {
            "target": "ca_target",
            "zR": "rn_factor"
          }
        }
      ]
    }
  }
}
```

### 参数

contract 区分 fit-only、apply-only 和 hybrid-only 参数。常见 self-renormalization 字段包括 `kernel_id`、`kernel_parameters`、`mu`、`LambdaQCD_gev`、`svdcut` 和 `z_coverage_policy`。fit 要求 reference operator 的 `d`；apply 要求 target operator 的 `d` 和 `m0_gev`。Hybrid application 还要求 `zs_fm` 和 denominator。

### 输出

- fit jobs 为 reusable factor 写出 `output.nc`、`diagnostics/self_renormalization.json` 以及 `plots/` 下的 fit panels。
- apply jobs 写出重整化后的 `output.nc`、`diagnostics/renormalization.json` 和结果图。
- stage report 链接每个已声明的 artifact，并总结 coverage、parameter provenance 和 fit quality。

## 开发

使用 uv 或 pip 安装开发依赖：

```bash
uv pip install -e ".[dev]"
# or
python -m pip install -e ".[dev]"
```

架构、文件所有权、测试和贡献者工作流记录在 [`DEVELOPMENT.md`](DEVELOPMENT.md) 中。

## 相关链接

- [LQCD_Master](https://github.com/sjtu-sai-agents/LQCD_Master)（[arXiv:2607.15001](https://arxiv.org/abs/2607.15001)）
