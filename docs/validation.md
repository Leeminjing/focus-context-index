# 首版交付验证

验证日期：2026-10-27（Asia/Shanghai）。Windows 本地运行；下列结果不包含真实 Provider 请求、Focus 数据库事务或完整 Patrol 验收。

| 检查 | 结果 |
|---|---|
| Python 3.13.13 / Pydantic 2.14.1 | `pytest -q -W error`：58 项通过 |
| Python 3.11.15 / Pydantic 2.10.6 | `pytest -q -W error`：58 项通过 |
| Ruff | 格式化与 lint 通过 |
| 原始 Focus 确定性 payload 与 hash | 固定 `08fe98c`，原始模块隔离执行，对照一致 |
| 完整索引接回原始 Focus 模型 | 原始 `RevisionSemanticIndex` 与局部 proof 校验通过 |
| 构建 | wheel 与 sdist 成功生成，包含许可证与来源说明 |
| 离线基准 | 3 类语料 × 3 个分组参数 × 2 次重复 × cold/append/exact，共 54 次，零失败 |

测试覆盖稳定段免重抽、追加边界、来源重新绑定、工具原子、独立 verdict、跨段否定纠错、累计阅读依赖、缓存篡改、命名空间／Context 隔离、预算竞争、未知用量、取消回收、构建合并、预热、SQLite 重启恢复、配置变更与失败报告分母。

离线基准报告在本地 `benchmark-results/smoke.json`；该目录被 Git 忽略，结果可由 README 命令重建。报告含实现指纹、benchmark 文件 hash、Python／Pydantic 环境、corpus hash、策略、随机执行顺序和每个样本。所有 exact 场景模型调用为零。脚本模型没有真实 token 用量，其 token 字段明确保留未知。

真实模型的语义质量、等待时间和消费还需要用同配置 Provider 重复实验，并进行人工语义评审。本轮只建立独立运行、兼容与实验基础，不提出生产提速比例。CI 已配置 Windows／Linux、Python 3.11–3.13；远程 CI 尚未运行。
