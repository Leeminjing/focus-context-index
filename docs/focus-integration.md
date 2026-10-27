# Focus 接入

本轮提供可运行示例与真实模型工厂接法，尚未修改 Focus 的 `PortfolioSemanticIndexService`。迁移时把它逐步收薄为宿主适配层，Portfolio catalog、Curator、Patrol 和数据库授权继续留在 Focus。

## 安装与模型

在 Focus 的 Python 环境安装独立库的本地目录，之后可以改为固定版本的私有包或 Git 提交依赖：

```powershell
python -m pip install -e D:\focus-context-index
```

[示例文件](../examples/focus_integration.py) 的 `make_focus_models(app_config)` 直接使用当前源码中的角色：

```python
models = FocusModelAdapter(
    projector_factory=lambda: RoleBoundStructuredModel(
        app_config, "semantic_index_projector", max_attempts=1
    ),
    verifier_factory=lambda: RoleBoundStructuredModel(app_config, "claim_verifier", max_attempts=1),
    interpreter_factory=lambda: RoleBoundStructuredModel(
        app_config, "semantic_index_interpreter", max_attempts=1
    ),
)
```

实际示例还把 `StructuredWorkerModel.estimate_input_tokens` 接入请求预检，并保留角色专属输出上限。必须使用 `max_attempts=1`，让引擎对每次验证重试分别准入与结算；宿主 transport 若内部重试，也必须报告所有尝试。

## 冻结输入与输出

宿主先按冻结来源身份读取已提交 Revision、检查观察到的内容 hash 与 delegation scope，再读取 `ContextRevisionReader.read(session, source, "semantic")`。完成读取后退出事务，再调用模型阶段：

```python
from focus_context_index import BuildAuthorization, IndexEngine, snapshot_from_focus

snapshot = snapshot_from_focus(
    namespace=workspace_or_tenant_id,
    revision_ref=revision.ref,
    content_hash=revision.content_hash,
    semantic_messages=semantic_view.messages,
    context_role=context_role,
    active_objective=objective,
)

result = await engine.build(
    snapshot=snapshot,
    baseline=validated_ancestor_artifact,
    authorization=BuildAuthorization(
        owner_id=loop_id,
        grant_id=f"{authority_revision}:{round_id}:{fencing_token}",
        allowed_context_ids=tuple(context_scope),
        allowed_namespaces=(workspace_or_tenant_id,),
    ),
    on_progress=report_progress,
)

# Focus 使用自己的类重新校验返回值，再按原发布规则提交。
index = FocusRevisionSemanticIndex.model_validate(result.index.model_dump(mode="json"))
```

这里的 `FocusRevisionSemanticIndex` 指 Focus 的原 `RevisionSemanticIndex` 类。`result.artifact` 应由宿主保存，用于下一版 baseline；`result.index` 是原 wire 格式。库中的 namespace 是缓存隔离域，必须由可信宿主确定；checkpoint 等字段只是冻结来源身份，不代表库自行获得读取权限。

`semantic_messages` 必须来自 semantic view。引擎保留 `semantic_policy`、`source_ordinal`、工具结构与元数据；它不负责从 execution view 猜测并排除控制／reasoning 消息。

## 必须接入的宿主端口

| 端口 | Focus 实现需要完成的工作 |
|---|---|
| `ModelAdapter` | 创建独立角色模型，提供完整配置身份和逐尝试用量 |
| `IndexStore` | 短事务查询局部 proof，原子保存赢家；发布完整 Index 前重验记录身份 |
| `Admission` | 当前授权、执行所有权、fencing、预算预留与幂等结算 |
| baseline 选择 | 已提交、同 Context、允许继承的演化链，原方案的有界祖先搜索 |
| Portfolio 适配层 | 失败转 blocker、进度持久化、索引／目录完整发布，再交 Curator |

默认 `InMemoryAdmission` 和 `SQLiteIndexStore` 用于独立研发。接到生产 Loop 时，把它们替换为 Focus 的短事务存储与授权预算实现。不要同时让旧 `_record_attempt_usage` 和新 `Admission.settle` 为同一模型调用重复计费；应以 call_id 对接宿主 receipt，并让批次汇总跳过已经托管的调用。

## 离线示例与兼容验证

```powershell
python -m examples.focus_integration --store index.sqlite --artifact benchmark-results/artifact.json
python tools/check_focus_compatibility.py --focus-root D:\path\to\Focus --artifact benchmark-results/artifact.json
```

兼容工具要求 Focus HEAD 恰好是 `08fe98c28896992c6195912e95b507599b9dcd76`，在隔离模块环境执行其原始索引、分组、grounding 与 proof 代码，校验确定性 payload／hash 和新完整产物。它不启动 Focus 应用，也不覆盖 PostgreSQL 发布事务、预算 fencing、完整 Patrol 流程或真实 Provider 验收。
