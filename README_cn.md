<h1 align="center">TSPilot: Agentic AI-native Time Series Data Interaction System</h1>

<p align="center"><b>面向时序数据库数据交互与分析的 AI 原生智能体系统</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.10 or later">
  <img src="https://img.shields.io/badge/TypeScript-5.3%2B-3178C6?style=flat-square&amp;logo=typescript&amp;logoColor=white" alt="TypeScript 5.3 or later">
  <img src="https://img.shields.io/badge/React-18-087EA4?style=flat-square&amp;logo=react&amp;logoColor=white" alt="React 18">
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&amp;logo=fastapi&amp;logoColor=white" alt="FastAPI 0.110 or later">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-F2C94C?style=flat-square" alt="MIT License"></a>
</p>

<p align="center">
  自然语言数据交互&nbsp;&nbsp;·&nbsp;&nbsp;多时序数据库接入&nbsp;&nbsp;·&nbsp;&nbsp;完整查询结果分析<br>
  可复用的分析产物&nbsp;&nbsp;·&nbsp;&nbsp;分析结论可视化验证
</p>

<p align="center">
  <a href="README.md">English</a>&nbsp;&nbsp;|&nbsp;&nbsp;<b>简体中文</b>
</p>

<p align="center">
  <a href="#-notification">Notification</a>&nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="#-产品界面">产品界面</a>&nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="#-核心能力">核心能力</a>&nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="#-快速开始">快速开始</a>&nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="#-数据库接入">数据库接入</a>&nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="#-项目文档">项目文档</a>
</p>

---

## 📣 Notification

- **【2026-08-20】** TSPilot v0.1 发布。

## 🧭 为什么需要 TSPilot

时序数据分析不只是编写一条数据库查询。用户通常还需要理解数据结构、时间范围、聚合方式和数据质量，并在不同的查询、分析和可视化工具之间反复切换，才能得到可靠的结论。

TSPilot 将自然语言交互与时序数据查询、分析和可视化连接起来，帮助用户完成数据探索、预测、异常检测和模式发现，并生成有数据依据的分析结论。

分析过程中产生的分析结论可以继续复用，让一次分析自然延伸为连续、深入的数据探索。

## 🖥️ 产品界面

<table align="center">
  <tr>
    <td align="center">
      <a href="tspilot-page.png"><img src="tspilot-page.png" alt="TSPilot time-series data interaction interface" width="1200"></a><br>
      <sub>通过统一界面连接时序数据，发起查询并完成连续的数据分析 · 点击查看大图</sub>
    </td>
  </tr>
</table>

<!-- <p align="center">
  <code>Connect</code>&nbsp;&nbsp;→&nbsp;&nbsp;<code>Ask</code>&nbsp;&nbsp;→&nbsp;&nbsp;<code>Analyze</code>&nbsp;&nbsp;→&nbsp;&nbsp;<code>Verify</code>
</p> -->

## ✨ 核心能力

| 特性 | 详细说明 |
|---|---|
| 💬 **自然语言数据交互** | 通过自然语言完成时序数据的查询、探索和分析，降低数据库查询与数据分析的使用门槛。 |
| 🗄️ **多种时序数据库接入** | 统一连接和管理不同类型的时序数据库，为用户提供一致的数据交互体验。 |
| 📊 **完整查询结果分析** | 面向查询返回的完整数据开展统计分析、预测、异常检测和模式发现，获得更全面、可靠的分析结果。 |
| ♻️ **可复用的分析产物** | 将查询结果、分析结论和可视化内容沉淀为可复用的分析产物，支持在后续问题和连续分析中继续引用。 |
| 📈 **分析结论可视化验证** | 通过图表展示趋势、异常、预测和关键指标，帮助用户直观理解并验证分析结论。 |

## 🚀 快速开始

### 1. 环境准备

- Python 3.10+
- Node.js 和 npm
- OpenAI-compatible 模型服务
- 可访问的时序数据库

### 2. 安装项目

```bash
git clone https://github.com/feilvvl/TSPilot.git
cd TSPilot

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

npm --prefix frontend install
```

### 3. 配置模型

从示例文件创建本地配置：

```bash
cp .env.example .env
```

如需从其他目录启动，可在 `.env` 中设置 `TSPILOT_ROOT=.`。随后打开前端的
**模型管理**，为模型填写模型名称、OpenAI-compatible API Base URL 与 API Key。
模型连接会保存到 `configs/models/ai/`，不再从 `.env` 读取模型凭据。

### 4. 连接数据库

数据库连接配置位于：

```text
configs/databases/
```

仓库当前提供 InfluxDB 和 Prometheus 的配置示例。复制对应示例并填写实际连接信息后，即可由 TSPilot 加载数据库资源。

### 5. 启动服务

```bash
BACKEND_PYTHON=.venv/bin/python scripts/dev.sh
```

默认访问地址：

- Web 界面：`http://localhost:5173`
- Backend API：`http://127.0.0.1:5680`
- API 文档：`http://127.0.0.1:5680/docs`

## 🗄️ 数据库接入

TSPilot 支持统一接入和管理多种时序数据库。完成数据库配置后，用户可以通过一致的自然语言交互方式查询和分析不同来源的时序数据。

<h3 align="center">🔌 目前支持的数据库</h3>

<p align="center"><sub>一次配置，以一致的方式连接、查询和分析不同来源的时序数据。</sub></p>

<table align="center">
  <tr>
    <td align="center"><img src="frontend/public/database-logos/influxdb.svg" alt="InfluxDB 2 and InfluxDB 3" height="42"><br><sub><b>InfluxDB 2 / 3</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/kdb.png" alt="kdb+" height="42"><br><sub><b>kdb+</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/prometheus.svg" alt="Prometheus" height="42"><br><sub><b>Prometheus</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/timescaledb.png" alt="TimescaleDB" height="42"><br><sub><b>TimescaleDB</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/dolphindb.png" alt="DolphinDB" height="42"><br><sub><b>DolphinDB</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/druid.png" alt="Apache Druid" height="42"><br><sub><b>Apache Druid</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="frontend/public/database-logos/questdb.png" alt="QuestDB" height="42"><br><sub><b>QuestDB</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/tdengine.png" alt="TDengine" height="42"><br><sub><b>TDengine</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/iotdb.svg" alt="Apache IoTDB" height="42"><br><sub><b>Apache IoTDB</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/victoriametrics.png" alt="VictoriaMetrics" height="42"><br><sub><b>VictoriaMetrics</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/griddb.png" alt="GridDB" height="42"><br><sub><b>GridDB</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/arc.svg" alt="Arc" height="42"><br><sub><b>Arc</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="frontend/public/database-logos/m3db.png" alt="M3DB" height="42"><br><sub><b>M3DB</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/cratedb.png" alt="CrateDB" height="42"><br><sub><b>CrateDB</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/cnosdb.png" alt="CnosDB" height="42"><br><sub><b>CnosDB</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/arcadedb.png" alt="ArcadeDB" height="42"><br><sub><b>ArcadeDB</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/greptimedb.png" alt="GreptimeDB" height="42"><br><sub><b>GreptimeDB</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/db2.jpg" alt="IBM Db2" height="42"><br><sub><b>IBM Db2</b></sub></td>
  </tr>
  <tr>
    <td align="center"><img src="frontend/public/database-logos/riak-ts.png" alt="Riak TS" height="42"><br><sub><b>Riak TS</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/bangdb.png" alt="BangDB" height="42"><br><sub><b>BangDB</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/machbase.png" alt="Machbase Neo" height="42"><br><sub><b>Machbase Neo</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/openmldb.jpg" alt="OpenMLDB" height="42"><br><sub><b>OpenMLDB</b></sub></td>
    <td align="center"><img src="frontend/public/database-logos/opengemini.png" alt="openGemini" height="42"><br><sub><b>openGemini</b></sub></td>
    <td align="center"><b>＋</b><br><sub><b>更多数据库</b></sub><br><sub>Soon</sub></td>
  </tr>
</table>

不同数据库的连接方式和运行依赖可能有所差异，具体配置请参考对应的数据库说明。

有关连接器接口和配置方式，请参阅：

- [数据库配置规范](configs/databases/SPEC/SPEC.md)
- [数据库模块规范](core/database/SPEC/SPEC.md)

## 📚 项目文档

| | 目标 | 文档 |
|:---:|---|---|
| 🧭 | 了解系统设计与能力边界 | [系统规范](TSPilot-v0.1-SPEC.md) |
| 📁 | 查看项目文件规范 | [文件规范](SPEC.md) |
| 🧩 | 了解模块职责边界 | [职责矩阵](RESPONSIBILITIES_MATRIX.md) |
| 🔗 | 查看 API 数据契约 | [API 规范](schemas/SPEC/api_SPEC.md) |
| 🗄️ | 接入时序数据库 | [数据库配置规范](configs/databases/SPEC/SPEC.md) |

## 🤝 参与贡献

欢迎提交 Issue 和 Pull Request。

涉及智能体行为、工具契约、数据库连接器或公共 API 的改动，应保持单一且明确的修改范围，补充相应测试，并同步更新相关规范文档。
