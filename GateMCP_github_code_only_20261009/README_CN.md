# GateMCP 论文对齐代码快照

本仓库快照包含论文命名版 GateMCP 源码、协议配置、测试和辅助工具。方法名称对齐版保留原控制器逻辑；本 GitHub 代码包仅不附带四份冻结基准数据和详细操作手册。

## 内容

- `qwen3_framework/general_mcp/`：控制器、任务适配器、MCP接口和指标。
- `qwen3_framework/scripts/`：单元运行、论文矩阵、结果汇总脚本。
- `config/`：论文协议和模型路径示例配置。
- `tests/`、`tools/`：测试、诊断和差分辅助工具。
- `METHODS.md`、`CHANGES.md`、`SOURCE_PROVENANCE.json`：方法映射、变更说明和源文件来源。

## 冻结评测数据

由于本包未确认所有数据集的再分发权，本 GitHub 代码包不包含四份冻结 JSONL 数据。所需文件名、数量和 SHA-256 见 `DATA_REQUIREMENTS.md`。请按原始数据条款获取数据，放入 `qwen3_framework/data/full_evaluation/`；运行器启动前会校验数据。

## 环境与运行

需要 Python 3.10 或更新版本。请在你管理的环境中安装依赖：

```sh
python -m pip install -r requirements.txt
```

按实际运行环境修改 `config/model_paths.example.json`，先查看 dry-run 计划：

```sh
python qwen3_framework/scripts/run_paper_matrix.py --model-key qwen3-1.7b
```

执行实验需要已获准使用的模型文件和通过哈希校验的四份数据，并且必须显式传入 `--execute`。

## 测试

可运行标准库测试：

```sh
python -m unittest discover -s tests -v
```

依赖冻结数据的测试在数据缺失时会跳过。本次打包没有运行测试套件。

## 许可证与第三方材料

本快照没有替项目指定许可证。公开发布为开源项目之前，请先阅读 `GITHUB_UPLOAD_NOTES_CN.md`，确认项目代码权属和第三方条款。数据仍受原始来源条款约束。