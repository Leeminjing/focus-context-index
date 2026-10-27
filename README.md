# focus-context-index

面向多 Context、多版本消息的增量语义索引 Python 库。目标是在相同语义质量要求下，减少 Patrol 等消费者等待索引的时间和模型消费。

首版从 [Focus `08fe98c`](https://github.com/Leeminjing/Focus/commit/08fe98c28896992c6195912e95b507599b9dcd76) 抽离，包含协议安全分组、局部抽取与独立验证、跨段整体理解、精确引文与阅读依赖凭据、增量缓存和异步调度。运行不需要 Focus、数据库服务或模型 SDK；真实模型由宿主注入。

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m examples.focus_integration
.venv\Scripts\python -m benchmarks.run --repeats 3 --segment-sizes 12 20 30
```

示例和默认 benchmark 使用脚本化模型，不发网络模型请求。它们验证协议、复用和消费记录，不能证明语义质量或真实模型提速。没有 Provider 用量时，token 字段为 `null`，不会假装为零。

```python
from focus_context_index import IndexEngine, IndexPolicy, SQLiteIndexStore

# models 实现 ModelAdapter；snapshot 是宿主提供的 FrozenSnapshot。
engine = IndexEngine(models=models, store=SQLiteIndexStore("index.sqlite"), concurrency=4)
result = await engine.build(snapshot=snapshot, policy=IndexPolicy())

next_result = await engine.build(snapshot=next_snapshot, baseline=result.artifact)
await engine.prewarm(snapshot=another_snapshot)  # 只完成并保存局部记录

index = next_result.index
stats = next_result.stats
```

`build()` 只返回带完整局部凭据和整体理解凭据的索引；引文或独立 verdict 不合格的命题进入拒绝账本，未覆盖的任务消息有明确的 hypothesis fallback。整体理解失败会抛出带 `stats` 的 `IndexBuildError`，不发布局部-only 产物。`prewarm()` 不发布完整索引。

同一个事件循环共用一个 Engine：所有阶段共享模型并发名额，前台优先，后台最多占一个名额；局部任务数有界。相同输入、模型合同和授权合并进行中的构建，一个等待者取消不影响其他人，最后一个等待者取消时回收子任务并结算。进度回调接收 `IndexProgress`，应快速返回。

稳定旧段保留原始凭据，组装时重新绑定到目标 Revision。每次内容变化都重新做整版理解；跨 Context 共享缓存、命题关系级增量解释、Context 公平队列与跨进程执行租约留作后续实验。SQLite 提供持久化赢家与重启恢复，不提供跨进程模型去重或预算权威。

- [Focus 接入与宿主职责](docs/focus-integration.md)
- [模块边界与兼容约束](docs/architecture.md)
- [实验方法与研发顺序](docs/benchmarks.md)
- [本轮验证记录](docs/validation.md)
- [来源文件与校验和](docs/upstream-manifest.json)

Apache-2.0。迁入代码及修改说明见 [NOTICE](NOTICE)。当前为研究版本，尚未发布到 PyPI，也未修改 Focus 的生产调用链。
