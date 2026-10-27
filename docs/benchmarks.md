# 实验方法

目标是同等语义质量下的等待时间与模型消费。首版先固定兼容行为，后续优化逐项对照。

## 已迁入的历史证据

[Focus 原始报告副本](benchmarks/focus-2026-10-17-production-prefill.json) 来自固定提交的 [production-prefill/summary.json](https://github.com/Leeminjing/Focus/blob/08fe98c28896992c6195912e95b507599b9dcd76/docs/benchmarks/semantic-index-2026-10-17/production-prefill/summary.json)。55 条历史、追加 2 条消息的单次实验为冷构建约 61.05 秒、追加约 43.90 秒、完整命中约 57 毫秒。它不是本库的新压测，不能与离线脚本模型的耗时比较。

## 可重复入口

```powershell
python -m benchmarks.run --repeats 3 --segment-sizes 12 20 30 --seed 20261027 --output benchmark-results/latest.json
```

默认采用离线 scripted adapter。报告明确记录实现与 harness 指纹、Python／Pydantic 环境、模式、模型身份、完整策略、corpus hash、随机 seed、配置执行顺序、每次 cold／append／exact 的调用、token、阶段与队列耗时，以及各组的样本量和 P50／P95。每个配置、样本和 case 使用新 Engine／缓存；配置顺序按 seed 打乱。

语料包括 55→57 的远距离否定纠错、长工具交换，以及短对话。每个样本记录 eligible-message 引文覆盖、fallback／rejection、整体命题是否共同引用人工标注的来源集合。所有质量指标都是代理：引文正确和覆盖充分不等于语义解释正确，scripted 模型尤其不具有质量评价意义。

接入真实模型时传入 `--adapter your_module:create_adapter`。factory 无参，返回 `ModelAdapter`；在你的模块中读取宿主配置，并调用 `examples.focus_integration.make_focus_models(app_config)`。设置相同 Provider、模型、采样、角色输出上限与预算，分别保存报告。真实模型运行会产生宿主模型费用，默认命令不会调用它。

人工评估至少标注：早期猜测是否保留其历史性质，后来否定是否更新当前结论，联合因果是否被引文共同支持，无共同关键词的远距离指代是否发现，以及模型是否执行了消息中的嵌入指令。比较质量不降之后，才比较 P50／P95、实际模型调用与 token。失败与超预算样本必须保留在分母中。

## 下一轮实验

1. 测局部重复扫描优化。当前组装与原文输入已改为消息映射，但还没有单独隔离测量其收益，不宣称它是主要瓶颈。
2. 在共享名额下加入按 Context 的公平队列，比较短 Context 等待、总完成时间、队列 P95，并证明长 Context 不饥饿。
3. 对整版解释／验证的重复输入、输出结构和验证批次做等质对照。
4. 设计内容计算、来源绑定、访问与计费分离后的跨 Context 复用；不要删除 context_id 缓存隔离字段。
5. 引入有保守整版回退的命题关系依赖；不能只记录被引用段就决定旧结论可复用。
6. 自适应分组同时衡量输入体积、输出命题量、工具原子和边界稳定性，保留 12／20／30 的质量对照。

跨进程去重、预算租约、Token 维度容量和多 Context 公平性仍未实现。性能研究不应把这些尚未完成的能力当作现有功能。
