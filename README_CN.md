# LogicCon

论文 **From Structured Facts to Logical Conflicts: A Benchmark and Framework for Visual-Text Conflict** 的推理方法代码。

[English](README.md) · [方法细节](docs/method.md) · [评测说明](docs/evaluation.md) · [配置说明](docs/configuration.md)

方法使用同一个视觉语言模型完成：目标感知的事实拆解 → 逐事实视觉验证 → 证据不足时补充验证 → 冲突聚合和类型诊断 → 最终回答。每条事实最多验证 **3 轮（包括首次验证）**，获得明确证据后提前停止。方法在推理时运行，不需要额外训练。

## 安装

需要 Python 3.11 或更高。使用 CUDA 12.6 的示例：

```bash
git clone https://github.com/changcv2021/LogicCon.git
cd LogicCon
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-inference.txt
python -m pip check
```

HTTP 后端、编排逻辑、指标和单元测试只依赖 Python 标准库，可单独 `pip install -e .`。

## 输入与运行

输入 JSONL 每行严格包含 `sample_id`、`image`、`statement` 三个字段。图片路径相对于 `--image-root`，不传标签、来源问题、场景图或事实变异记录。

```json
{"sample_id":"example-001","image":"images/photo.jpg","statement":"The cup to the left of the plate is red."}
```

[LogicCon 数据集](https://huggingface.co/datasets/benchmarkanon/logic_conflict) 的离线包可直接使用其 `inputs/test.jsonl` 和图片根目录。

```bash
export LOGICCON_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct

logiccon-reason validate-config --config configs/reasoning/qwen2_5_vl_7b.toml

logiccon-reason infer \
  --config configs/reasoning/qwen2_5_vl_7b.toml \
  --inputs /path/to/LogicCon/inputs/test.jsonl \
  --image-root /path/to/LogicCon \
  --output runs/qwen7b-full --mode full
```

模型默认从本地加载。另有 Qwen2.5-VL-32B、HF LLaVA-1.5 和兼容 HTTP 服务的配置。共享 HPC 上请使用获准的 GPU allocation。

调试加 `--limit 4`；同一实验续跑加 `--resume`。不同设置使用不同输出目录。`--limit` 只取输入前缀，不是随机抽样；正式评测时预测和 gold 范围需要一致。

## 消融与输出

`--mode` 可选 `base`、`parsing`、`verification`、`aggregation`、`full`，依次增加论文中的模块。完整解释见英文 README 的对应表格。

输出包括预测 `predictions.jsonl`、逐调用记录 `events.jsonl`、复现元信息 `manifest.json` 和完成情况 `summary.json`。保留每轮证据、模型原始回答、调用次数、token 数、错误和不确定性。

需要并行时设置相同 `--num-shards`、不同 `--shard-index`，每个进程使用独立目录；结束后用 `logiccon-reason merge` 合并全部分片。

## 评测与测试

```bash
logiccon-reason score \
  --gold /path/to/LogicCon/annotations/test.jsonl \
  --predictions runs/qwen7b-full/predictions.jsonl \
  --output runs/qwen7b-full/metrics.json

python -m unittest discover -s tests -v
```

检测与类型指标可直接计算。TO、TC、VE、CP 需要独立语义 judge，使用 `judge` 命令生成判分文件后，通过 `score --judgments` 传入；缺少判分不会用字符串匹配代替。CP 要求同一样本三个分量全部正确。详见 [评测文档](docs/evaluation.md)。

引用见 [CITATION.cff](CITATION.cff)，许可见 [LICENSE](LICENSE)，上游材料说明见 [NOTICE.md](NOTICE.md)。
