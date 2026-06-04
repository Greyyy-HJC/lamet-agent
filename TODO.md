# TODO: LaMET Agent 项目内容与分工

本文件用于整理 LaMET Agent 项目的主要开发内容、理想项目框架和测试计划，便于和合作者讨论任务拆分。

## 0. 项目目标

构建一个面向 LaMET / LQCD 数据分析流程的 agent 原型。该 agent 应该能够：

- 根据用户给定的数据、物理输入和分析目标，组织完整的数据分析 workflow。
- 在需要物理判断的环节辅助选择分析策略，例如关联函数拟合、矩阵元提取、外推策略选择。
- 在相对固定的环节调用确定性的 Python 工具完成计算，例如重整化、傅里叶变换、微扰匹配。
- 保留重采样样本，系统地传播统计误差。
- 扫描合理分析方案，估计系统误差。
- 输出中间结果、诊断图、最终结果和可读分析报告。

---

## 1. 需要写的内容分类

### 1.1 总控 Agent / Runtime

需要实现一个主程序，用于调用 LLM API，并接入总控 prompt。

TODO:

- [ ] 编写 `main.py`，作为命令行入口。
- [ ] 编写 `orchestrator.py`，实现 agent 主循环。
- [ ] 编写 `llm_client.py`，封装 OpenAI / Anthropic / DeepSeek 等 LLM API。
- [ ] 支持读取用户配置文件，例如 `config.yaml`。
- [ ] 支持读取 workflow state，例如 `state.json`。
- [ ] 支持结构化 action 输出，例如 JSON schema。
- [ ] 支持工具调用结果回传给 LLM。
- [ ] 支持暂停、恢复和重新运行某一步。
- [ ] 支持保存完整日志，例如 `logs.jsonl`。

建议原则：

- LLM 不直接操作底层数组。
- LLM 只生成结构化 action。
- 所有数值计算都通过固定 Python 函数完成。
- 每一步的输入、输出、诊断信息都必须落盘。

---

### 1.2 Prompt 系统

需要将 prompt 分成总控 prompt 和各流程 skill prompt。

TODO:

- [ ] 编写 `prompts/system.md`：定义 agent 的总体身份、边界和输出规范。
- [ ] 编写 `prompts/controller.md`：定义总控逻辑、workflow 编排方式、工具调用格式。
- [ ] 编写 `prompts/skills/correlator_analysis.md`。
- [ ] 编写 `prompts/skills/renormalization.md`。
- [ ] 编写 `prompts/skills/fourier_transform.md`。
- [ ] 编写 `prompts/skills/perturbative_matching.md`。
- [ ] 编写 `prompts/skills/extrapolation.md`。
- [ ] 为每个 skill 规定输入检查清单。
- [ ] 为每个 skill 规定输出 JSON schema。
- [ ] 明确哪些步骤允许 agent 自主选择策略。
- [ ] 明确哪些步骤必须由用户指定物理方案。

建议分工：

- Correlator analysis：策略型 prompt，允许 agent 辅助选择拟合方案、拟合区间、prior、model averaging 策略。
- Renormalization：执行型 prompt，用户指定 scheme，agent 检查输入并调用工具。
- Fourier transform：半执行型 prompt，agent 可辅助选择 asymptotic extrapolation 区间，但 transform convention 必须明确。
- Perturbative matching：执行型 prompt，用户指定 kernel、scheme、scale 和 convention。
- Extrapolation：策略型 prompt，允许 agent 辅助比较不同连续极限、手征极限、无穷体积外推 ansatz。

---

### 1.3 数据结构 / Schema

需要定义统一的数据对象，保证不同步骤可以稳定衔接。

TODO:

- [ ] 定义 `ResampleArray`，保存 mean、samples、covariance、resampling method。
- [ ] 定义 `CorrelatorDataset`，保存 2pt / 3pt 关联函数、时间变量、动量、算符信息。
- [ ] 定义 `FitSpec`，保存拟合函数、拟合区间、prior、covariance 处理方式。
- [ ] 定义 `FitResult`，保存参数、误差、chi2/dof、Q value、AIC/BIC、诊断信息。
- [ ] 定义 `MatrixElement`，保存裸矩阵元及其 metadata。
- [ ] 定义 `RenormSpec`，保存重整化方案、常数、参考矩阵元、scheme、scale。
- [ ] 定义 `CoordinateMatrixElement`，保存坐标空间矩阵元 h(z, Pz)。
- [ ] 定义 `MomentumDistribution`，保存 quasi distribution 或其他动量空间结果。
- [ ] 定义 `MatchingSpec`，保存 matching kernel、scheme、scale、x-grid。
- [ ] 定义 `ExtrapolationDataset`，保存不同 ensemble 的物理量、a、m_pi、L 等信息。
- [ ] 定义 `AnalysisState`，保存整个 workflow 的状态。
- [ ] 定义 `AnalysisReport`，保存最终报告所需内容。

关键 metadata:

- ensemble id
- lattice spacing `a`
- volume `L^3 x T`
- pion mass `m_pi`
- momentum `Pz`
- operator definition
- source-sink separation
- insertion time
- z direction convention
- renormalization scheme
- matching scheme
- scale `mu`
- resampling method
- unit convention

---

### 1.4 固定 Python 工具库

需要将常用分析步骤写成稳定函数。agent 只负责调用这些函数，不临时生成核心算法。

#### 1.4.1 I/O 和基础工具

TODO:

- [ ] HDF5 / npy / csv / yaml 读写。
- [ ] 数据格式检查。
- [ ] 单位转换。
- [ ] metadata 校验。
- [ ] 结果落盘。
- [ ] 自动生成目录结构。

#### 1.4.2 Resampling 和误差传播

TODO:

- [ ] Bootstrap。
- [ ] Jackknife。
- [ ] Blocking / binning。
- [ ] 协方差矩阵估计。
- [ ] shrinkage covariance。
- [ ] SVD cut。
- [ ] correlated / uncorrelated error propagation。

#### 1.4.3 Correlator analysis

TODO:

- [ ] 2pt effective mass。
- [ ] 2pt 单态 / 多态拟合。
- [ ] 3pt ratio 构造。
- [ ] plateau fit。
- [ ] summation method。
- [ ] 2pt + 3pt simultaneous fit。
- [ ] GEVP。
- [ ] model averaging。
- [ ] excited-state contamination 诊断。
- [ ] fit window scan。
- [ ] prior sensitivity scan。
- [ ] 输出裸矩阵元。

#### 1.4.4 Renormalization

TODO:

- [ ] Constant renormalization。
- [ ] Ratio renormalization。
- [ ] Hybrid scheme。
- [ ] Self-renormalization 拟合。
- [ ] `m0` matching。
- [ ] 构造 `M_R(z)`。
- [ ] 抽取 `Z_R(z, a)`。
- [ ] 检查重整化后残余格距依赖。
- [ ] 保留 resampling samples。

#### 1.4.5 Fourier transform

TODO:

- [ ] z-space 数据对称化 / 反对称化。
- [ ] 实部和虚部联合处理。
- [ ] asymptotic extrapolation 拟合。
- [ ] GI method quasi-PDF extrapolation ansatz。
- [ ] CG method quasi-PDF extrapolation ansatz。
- [ ] zmin / zmax scan。
- [ ] smoothing function scan。
- [ ] finite-z truncation 诊断。
- [ ] Fourier transform 到 x-space。
- [ ] 输出 quasi distribution。

#### 1.4.6 Perturbative matching

TODO:

- [ ] matching kernel 接口。
- [ ] scheme conversion。
- [ ] scale evolution。
- [ ] convolution implementation。
- [ ] plus distribution 数值处理。
- [ ] x-grid interpolation。
- [ ] endpoint behavior 处理。
- [ ] matching 后误差传播。

#### 1.4.7 Extrapolation

TODO:

- [ ] continuum extrapolation ansatz。
- [ ] chiral extrapolation ansatz。
- [ ] finite-volume correction ansatz。
- [ ] combined global fit。
- [ ] Bayesian prior 支持。
- [ ] model averaging。
- [ ] fit range / ensemble selection scan。
- [ ] systematic uncertainty estimate。
- [ ] physical point prediction。

#### 1.4.8 Plotting 和报告

TODO:

- [ ] effective mass plot。
- [ ] ratio plot。
- [ ] fit stability plot。
- [ ] posterior / prior comparison plot。
- [ ] renormalization factor plot。
- [ ] z-space matrix element plot。
- [ ] asymptotic extrapolation plot。
- [ ] quasi distribution plot。
- [ ] matched distribution plot。
- [ ] continuum/chiral/volume extrapolation plot。
- [ ] 自动生成 Markdown / LaTeX 报告。

---

## 2. 理想项目框架

建议目录结构如下：

```text
lamet_agent/
  pyproject.toml
  README.md
  TODO.md
  configs/
    example_ga.yaml
    example_pdf.yaml
    example_tmd.yaml

  lamet_agent/
    __init__.py

    main.py
    orchestrator.py
    llm_client.py
    tool_registry.py
    workflow.py
    logging_utils.py

    prompts/
      system.md
      controller.md
      skills/
        correlator_analysis.md
        renormalization.md
        fourier_transform.md
        perturbative_matching.md
        extrapolation.md

    schemas/
      __init__.py
      state.py
      actions.py
      datasets.py
      fit.py
      renormalization.py
      fourier.py
      matching.py
      extrapolation.py
      report.py

    tools/
      __init__.py
      io.py
      validation.py
      resampling.py
      covariance.py
      fitting.py
      correlator.py
      gevp.py
      renormalization.py
      fourier.py
      matching.py
      extrapolation.py
      model_averaging.py
      plotting.py
      report.py

    workflows/
      __init__.py
      correlator_to_matrix_element.py
      renorm_to_quasi.py
      quasi_to_matched.py
      full_lamet_pipeline.py

  tests/
    unit/
      test_resampling.py
      test_covariance.py
      test_fitting.py
      test_correlator.py
      test_renormalization.py
      test_fourier.py
      test_matching.py
      test_extrapolation.py
      test_schemas.py

    integration/
      test_correlator_pipeline.py
      test_renorm_fourier_pipeline.py
      test_full_mock_lamet_pipeline.py
      test_resume_workflow.py

    regression/
      test_known_ga_result.py
      test_known_quasi_pdf_result.py

    prompts/
      test_action_schema.py
      test_controller_prompt.py
      test_skill_prompt_outputs.py

  examples/
    ga_analysis/
      config.yaml
      input_data/
      expected_outputs/

    quasi_pdf_analysis/
      config.yaml
      input_data/
      expected_outputs/

  notebooks/
    demo_correlator_analysis.ipynb
    demo_fourier_transform.ipynb

  runs/
    .gitkeep
```

---

## 3. Workflow 设计

### 3.1 总体 workflow

```text
User config + data
      |
      v
Input validation
      |
      v
Controller agent
      |
      +--> Correlator analysis skill
      |       |
      |       v
      |   bare matrix element
      |
      +--> Renormalization skill
      |       |
      |       v
      |   renormalized coordinate-space matrix element
      |
      +--> Fourier transform skill
      |       |
      |       v
      |   quasi distribution
      |
      +--> Perturbative matching skill
      |       |
      |       v
      |   matched distribution
      |
      +--> Extrapolation skill
              |
              v
          physical result + report
```

### 3.2 Agent action 类型

建议限定 agent 只能输出以下 action：

```text
- inspect_input
- call_tool
- compare_results
- update_state
- request_user_input
- generate_report
- finish
```

其中最重要的是 `call_tool`：

```json
{
  "action": "call_tool",
  "tool_name": "fit_2pt_multistate",
  "args": {
    "dataset_id": "proton_2pt_Pz0",
    "tmin": 5,
    "tmax": 14,
    "n_states": 2,
    "covariance": "shrinkage"
  },
  "reason": "Two-state fit is needed to estimate excited-state contamination."
}
```

### 3.3 Workflow state

建议每次工具调用后更新 `state.json`：

```json
{
  "run_id": "2026-xx-xx_ga_demo",
  "current_step": "correlator_analysis",
  "completed_steps": [],
  "available_datasets": [],
  "tool_results": [],
  "diagnostics": [],
  "warnings": [],
  "final_outputs": {}
}
```

---

## 4. Tests

### 4.1 Unit tests

#### Resampling

- [ ] Bootstrap mean and error agree with analytic expectation for Gaussian mock data.
- [ ] Jackknife error agrees with analytic expectation for simple observables.
- [ ] Blocking reduces autocorrelation for correlated mock chain.
- [ ] Resampling axis is preserved after arithmetic operations.

#### Covariance

- [ ] Sample covariance has correct shape and symmetry.
- [ ] Correlation matrix diagonal is one.
- [ ] SVD cut removes small eigenmodes correctly.
- [ ] Shrinkage covariance remains positive definite.

#### Fitting

- [ ] Single exponential fit recovers mock energy.
- [ ] Two-state fit recovers mock ground state within uncertainty.
- [ ] Correlated chi2 is computed correctly.
- [ ] Bayesian prior contribution is included correctly.
- [ ] Fit result stores chi2/dof, Q value and parameter covariance.

#### Correlator analysis

- [ ] Effective mass plateau is found for mock 2pt data.
- [ ] Plateau fit recovers mock matrix element.
- [ ] Summation method slope recovers mock matrix element.
- [ ] Simultaneous 2pt + 3pt fit recovers mock matrix element.
- [ ] Fit window scan ranks stable windows correctly.
- [ ] Model averaging gives sensible weighted result.

#### Renormalization

- [ ] Constant renormalization multiplies samples correctly.
- [ ] Ratio renormalization propagates correlated errors correctly.
- [ ] Hybrid scheme applies different schemes in short- and long-distance regions.
- [ ] Self-renormalization mock fit recovers injected parameters.
- [ ] `Z_R(z, a)` reconstruction satisfies `M(z, a) / Z_R(z, a) = M_R(z)` within tolerance.

#### Fourier transform

- [ ] Real part is symmetrized correctly.
- [ ] Imaginary part is anti-symmetrized correctly.
- [ ] Known Gaussian in z-space transforms to expected Gaussian in momentum space.
- [ ] Finite-z truncation behavior is captured.
- [ ] Asymptotic extrapolation recovers injected long-distance tail.
- [ ] zmin / zmax scan returns stable central values for mock data.

#### Matching

- [ ] Identity kernel returns the input distribution.
- [ ] Simple convolution with known kernel matches analytic result.
- [ ] Plus distribution numerical implementation passes normalization checks.
- [ ] Scheme conversion is applied consistently.
- [ ] x-grid interpolation preserves normalization within tolerance.

#### Extrapolation

- [ ] Linear `a^2` continuum extrapolation recovers physical limit.
- [ ] Combined `a^2 + m_pi^2 + exp(-m_pi L)` fit recovers mock physical value.
- [ ] Bayesian priors affect underconstrained fits as expected.
- [ ] Model averaging includes systematic spread.
- [ ] Removing one ensemble does not crash the pipeline.

#### Schemas

- [ ] All schema objects serialize to JSON.
- [ ] All schema objects deserialize from JSON.
- [ ] Required metadata fields are enforced.
- [ ] Unit conventions are validated.
- [ ] Invalid workflow states are rejected.

---

### 4.2 Integration tests

- [ ] Mock correlator pipeline: 2pt + 3pt data -> bare matrix element.
- [ ] Mock renormalization pipeline: bare matrix element -> renormalized matrix element.
- [ ] Mock Fourier pipeline: h(z, Pz) -> quasi distribution.
- [ ] Mock matching pipeline: quasi distribution -> matched distribution.
- [ ] Mock extrapolation pipeline: ensemble results -> physical point result.
- [ ] Full mock LaMET pipeline runs from config to report.
- [ ] Workflow can stop after one step and resume from `state.json`.
- [ ] Failed tool call produces recoverable error state.
- [ ] Missing required input triggers `request_user_input` action.

---

### 4.3 Prompt / Agent tests

这些测试不检查物理数值，而是检查 agent 是否遵守协议。

- [ ] Controller prompt always returns valid JSON action.
- [ ] Agent does not call undefined tools.
- [ ] Agent does not invent missing renormalization scheme.
- [ ] Agent asks for missing required inputs when necessary.
- [ ] Agent explains why it chooses a fit window.
- [ ] Agent compares multiple fit strategies before selecting final result.
- [ ] Agent records warnings when data quality is poor.
- [ ] Agent does not silently discard resampling samples.
- [ ] Agent produces reproducible action sequence when temperature is zero.

---

### 4.4 Regression tests

- [ ] Reproduce a known mock `g_A` analysis result.
- [ ] Reproduce a known quasi-PDF Fourier transform result.
- [ ] Reproduce a known renormalization factor on synthetic data.
- [ ] Reproduce a known extrapolated physical value on synthetic ensemble data.
- [ ] Ensure changes to tools do not alter previous benchmark outputs beyond tolerance.

---

## 5. Suggested milestone plan

### Milestone 1: Minimal agent runtime

- [ ] `main.py`
- [ ] `orchestrator.py`
- [ ] `llm_client.py`
- [ ] action schema
- [ ] tool registry
- [ ] state save / resume
- [ ] simple mock tool call

Goal: LLM 能稳定输出 action，并调用 Python 工具。

### Milestone 2: Correlator analysis prototype

- [ ] mock 2pt / 3pt generator
- [ ] effective mass
- [ ] plateau fit
- [ ] summation method
- [ ] simultaneous fit
- [ ] fit window scan
- [ ] basic report

Goal: 用 mock data 跑通 `2pt + 3pt -> bare matrix element`。

### Milestone 3: Renormalization + Fourier prototype

- [ ] constant renormalization
- [ ] ratio renormalization
- [ ] z-space convention handling
- [ ] asymptotic extrapolation
- [ ] Fourier transform
- [ ] quasi distribution output

Goal: 跑通 `bare matrix element -> renormalized h(z) -> quasi distribution`。

### Milestone 4: Matching + Extrapolation prototype

- [ ] identity / simple matching kernel
- [ ] convolution framework
- [ ] continuum extrapolation
- [ ] chiral / finite-volume terms
- [ ] model averaging

Goal: 跑通简化版 end-to-end LaMET workflow。

### Milestone 5: Robustness and documentation

- [ ] 完整 tests
- [ ] example configs
- [ ] example notebooks
- [ ] user documentation
- [ ] developer documentation
- [ ] benchmark results

Goal: 形成可以交给合作者使用和扩展的原型。

---

## 6. 建议合作者分工

### Person A: Agent runtime / prompt / orchestration

负责：

- LLM API 接入。
- 总控 prompt。
- action schema。
- workflow state。
- tool registry。
- 日志和报告生成。

### Person B: Correlator analysis

负责：

- 2pt / 3pt 拟合工具。
- GEVP。
- ratio / plateau / summation / simultaneous fit。
- fit window scan。
- excited-state contamination 诊断。
- mock correlator tests。

### Person C: Renormalization + Fourier transform

负责：

- constant / ratio / hybrid / self-renormalization。
- z-space convention。
- asymptotic extrapolation。
- Fourier transform。
- quasi distribution 输出。

### Person D: Matching + Extrapolation

负责：

- matching kernel 接口。
- convolution。
- scheme conversion。
- continuum / chiral / finite-volume extrapolation。
- model averaging。

### Person E: Testing / examples / documentation

负责：

- mock data。
- unit tests。
- integration tests。
- regression tests。
- example configs。
- README 和教程。

实际人数较少时，可以合并为三组：

```text
Group 1: Agent runtime + prompt + schema
Group 2: Correlator analysis + extrapolation
Group 3: Renormalization + Fourier + matching + tests
```

---

## 7. Open questions for meeting

- [ ] 第一版优先做 `g_A`，还是 quasi-PDF / LaMET 主流程？
- [ ] 数据格式是否统一成 HDF5？
- [ ] 是否要求兼容现有分析代码？
- [ ] 第一版支持 OpenAI API 还是同时支持 Claude / local model？
- [ ] 是否允许 agent 自动选择 renormalization scheme，还是必须完全由用户指定？
- [ ] prompt 是否作为纯 markdown 管理，还是做成 Python package 内置资源？
- [ ] report 输出 Markdown、LaTeX，还是 Jupyter notebook？
- [ ] tests 以 synthetic data 为主，还是尽快接入真实数据？
- [ ] 是否需要从第一版开始支持 parallel / batch runs？
- [ ] 是否需要把每一步做成可独立命令行工具？

