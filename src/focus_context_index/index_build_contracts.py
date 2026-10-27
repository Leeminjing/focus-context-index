# Derived from Leeminjing/Focus, Apache-2.0, commit 08fe98c28896992c6195912e95b507599b9dcd76.
# See NOTICE and docs/upstream-manifest.json for provenance and modifications.
r"""本文件对外提供 IndexInheritanceReceipt 与 IndexBuildPlan。

输入为冻结 segment inventories、基线 index 和消息前缀证明；输出为不可变构建计划和可审计 receipt。
具体工作流为检查复用／重算库存不交叠、增量必须有基线、全量必须有原因，再由完整 index 绑定记录 identities。
示例：plan = IndexBuildPlan(recomputed_segment_ids=ids, full_build_reason="no_baseline")。
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class IndexInheritanceReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["full", "incremental"] = "full"
    baseline_index_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prefix_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reused_segment_ids: tuple[str, ...] = ()
    recomputed_segment_ids: tuple[str, ...] = ()
    full_build_reason: str | None = "no_baseline"
    record_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_valid_plan(self) -> Self:
        reused, recomputed = self.reused_segment_ids, self.recomputed_segment_ids
        if len(set(reused + recomputed)) != len(reused) + len(recomputed):
            raise ValueError("index build segment inventory 重复或交叠")
        if self.mode == "incremental":
            if not self.baseline_index_id or not self.prefix_hash or self.full_build_reason:
                raise ValueError("incremental index 缺少基线／前缀证明")
        elif self.baseline_index_id or self.prefix_hash or reused or not self.full_build_reason:
            raise ValueError("full index 不得声称继承")
        return self


class IndexBuildPlan(IndexInheritanceReceipt):
    pass
