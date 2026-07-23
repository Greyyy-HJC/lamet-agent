# LaMET 文章库使用说明

## 1. 这个库是什么

这个目录维护的是一个本地 LaMET arXiv 文章知识库。它不镜像全部 arXiv PDF，而是先抓取和保存文章元数据，再按 LaMET 相关性打分，保留 `core` 和 `secondary` 两类文章。

当前知识库主要服务三类任务：

- 追踪 LaMET 及其相邻方法文献
- 本地检索某一主题、年份、标签下的文章
- 给后续的 agent 或 review 流程提供结构化文献背景

## 2. 最重要的文件

如果只看“所有爬取信息最核心存在哪里”，最重要的文件是：

- `data/lamet_arxiv.sqlite3`

这是主数据库。已经抓到并保留下来的文章元数据、标签、分数、标题、摘要、作者、日期、查询命中信息都在这里。

其他重要文件：

- `data/papers.jsonl`
  这是导出的 JSONL 快照，适合做后处理、向量化、外部脚本读取。
- `config/relevance_config.json`
  这是知识库的抓取与筛选规则，包括 query groups、关键词、权重、阈值。
- `config/manual_seeds.json`
  这是手工种子文章列表。你可以把确认重要的 arXiv ID 放进去，提升它们被保留的优先级。

如果你问“以后备份时最不能丢的是哪个文件”，优先级通常是：

1. `data/lamet_arxiv.sqlite3`
2. `config/relevance_config.json`
3. `config/manual_seeds.json`
4. `data/papers.jsonl`

## 3. 常用命令

在本目录下运行：

```bash
cd /Users/zhaodianjun/Desktop/lamet-papers
```

### 3.1 查看当前库状态

```bash
python3 scripts/harvest_lamet.py report
```

它会显示：

- 已抓到多少篇 `core`
- 已抓到多少篇 `secondary`
- 当前总数
- `bootstrap_progress_date`
- `last_harvest_date`
- 当前最晚文章日期

### 3.2 列出文章

列出最近 50 篇：

```bash
python3 scripts/harvest_lamet.py list --limit 50
```

只看核心文献：

```bash
python3 scripts/harvest_lamet.py list --label core --limit 50
```

只看次相关文献：

```bash
python3 scripts/harvest_lamet.py list --label secondary --limit 50
```

### 3.3 搜索文章

按关键词搜索标题和摘要：

```bash
python3 scripts/harvest_lamet.py search --query matching --limit 30
python3 scripts/harvest_lamet.py search --query factorization --limit 30
python3 scripts/harvest_lamet.py search --query nnlo --limit 30
python3 scripts/harvest_lamet.py search --query "collins-soper" --limit 30
```

按年份搜索：

```bash
python3 scripts/harvest_lamet.py search --query lamet --year 2017 --limit 30
```

### 3.4 导出 JSONL

```bash
python3 scripts/harvest_lamet.py export
```

导出结果默认写到：

- `data/papers.jsonl`

## 4. 如何抓取、续跑、回填

### 4.1 首轮主线抓取

```bash
python3 scripts/harvest_lamet.py bootstrap --end-date 2026-07-22 --page-size 3 --sleep-seconds 15 --window-days 10
```

这个命令适合长时间跑。若中途被 arXiv 限流或断连，下次直接重跑同一条命令即可，它会根据断点继续，不需要从头来。

### 4.2 增量更新到当前日期

当首轮抓取完成后，用：

```bash
python3 scripts/update_since_last_harvest.py
```

它会从上次最终爬取日期附近重新补抓一小段，然后更新到当前日期。

### 4.3 回填更早的年份但不覆盖已有结果

例如补 `2010-01-01` 到 `2014-04-25`：

```bash
python3 scripts/harvest_lamet.py backfill --start-date 2010-01-01 --end-date 2014-04-25 --page-size 3 --sleep-seconds 15 --window-days 10
```

这个命令不会清空已抓到的记录。已有文章只会按 `arxiv_id` 去重或更新。

## 5. 如何把 LaMET 微扰论工作也纳入

如果你希望把 LaMET 的微扰 matching、factorization、kernel、NLO/NNLO、anomalous dimension、Collins-Soper kernel 等工作纳入，这个库现在已经做了规则扩展，重点在：

- `config/relevance_config.json`

里面已经加入：

- `lamet_perturbative` query group
- `hep-th` 分类支持
- 更强的 perturbative关键词模式

但改规则之后，历史上此前未被纳入的文章不会自动出现。你需要把相关时间段重新抓一遍，例如：

```bash
python3 scripts/harvest_lamet.py bootstrap --start-date 2010-01-01 --end-date 2026-07-22 --page-size 3 --sleep-seconds 15 --window-days 10 --no-resume
```

如果你不想全范围重扫，也可以只回填早期缺口。

## 6. 我该如何真正“使用”这个文章库

最实用的用法有三种：

### 6.1 文献清单模式

先用 `list` 看最近文章，再用 `search` 缩小到具体主题，例如：

```bash
python3 scripts/harvest_lamet.py search --query matching --limit 50
python3 scripts/harvest_lamet.py search --query quasi --year 2017 --limit 50
python3 scripts/harvest_lamet.py search --query renormalization --limit 50
```

### 6.2 主题追踪模式

你可以定期运行：

```bash
python3 scripts/update_since_last_harvest.py
python3 scripts/harvest_lamet.py search --query lamet --limit 20
python3 scripts/harvest_lamet.py search --query nnlo --limit 20
```

这样可以快速发现最新的 LaMET 或微扰相关文章。

### 6.3 外部程序读取模式

如果要给别的程序或 agent 用，推荐直接读：

- `data/lamet_arxiv.sqlite3`
  适合结构化查询
- `data/papers.jsonl`
  适合批处理、向量化、RAG、脚本串联

