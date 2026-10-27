# Derived from Leeminjing/Focus, Apache-2.0, commit 08fe98c28896992c6195912e95b507599b9dcd76.
# See NOTICE and docs/upstream-manifest.json for provenance and modifications.
r"""本文件对外提供 FrozenInterpretationInputs。

输入为完整冻结 Index 和依次覆盖库存的局部 records；输出为全局发现目录、当前共同阅读窗口及完整累积阅读依赖。
具体工作流为保存全部有序目录和协议闭合原文，纯预检 read scope／上限，再替换活动窗口并累积实际提供与请求身份；模型 payload 只带活动窗口与最近读取身份，最终 proof 使用完整审计。
示例：inputs.read((early_segment_id, late_segment_id))；不以关键词、相邻窗口或当前 Context pointer 筛选来源。
"""

from .sizing import exceeds_limit

from .errors import IndexBudgetExceeded
from .interpretation_record import (
    InterpretationHint,
    InterpretationInventoryEntry,
    InterpretationOriginalSegment,
)


class FrozenInterpretationInputs:
    def __init__(self, index, records, *, max_reads):
        if len(index.segments) != len(records):
            raise ValueError("interpretation local inventory incomplete")
        self._segments = {}
        self._provided = {}
        self._window = ()
        self._requests = []
        self._max_reads = max_reads
        self.inventory = tuple(
            self._entry(segment, record)
            for segment, record in zip(index.segments, records, strict=True)
        )
        messages_by_id = {m.message_id: m.model_dump(mode="json") for m in index.messages}
        for segment in index.segments:
            self._segments[segment.segment_id] = InterpretationOriginalSegment(
                segment_id=segment.segment_id,
                messages=tuple(messages_by_id[identity] for identity in segment.message_ids),
            )

    @staticmethod
    def _entry(segment, record):
        accepted = set(record.accepted_claim_keys)
        return InterpretationInventoryEntry(
            segment_id=segment.segment_id,
            ordinal=segment.ordinal,
            message_ids=segment.message_ids,
            content_hash=segment.content_hash,
            descriptor=segment.descriptor,
            record_id=record.record_id,
            claims=tuple(
                InterpretationHint(
                    authority=d.authority,
                    statement=d.statement,
                    message_ids=tuple(s.message_id for s in d.supports),
                )
                for d in record.drafts
                if d.claim_key in accepted
            ),
            rejections=record.rejections,
            fallback=record.fallback,
        )

    @property
    def provided(self):
        return tuple(item for key, item in self._segments.items() if key in self._provided)

    @property
    def requests(self):
        return tuple(self._requests)

    def all_originals(self):
        return tuple(self._segments.values())

    def read(self, segment_ids, *, requested=True):
        selected = self._select(segment_ids)
        if requested:
            self._requests.append(tuple(segment_ids))
        self._window = selected
        self._provided.update({item.segment_id: item for item in selected})

    def preview_read(self, segment_ids, *, requested=True):
        selected = self._select(segment_ids)
        requests = (tuple(segment_ids),) if requested else ()
        return self._payload(selected, requests)

    def _select(self, segment_ids):
        requested = set(segment_ids)
        if len(requested) != len(segment_ids):
            raise ValueError("interpretation duplicate read identity")
        if not requested.issubset(self._segments):
            raise ValueError("interpretation read outside frozen scope")
        combined = set(self._provided) | requested
        if exceeds_limit(len(combined), self._max_reads):
            raise IndexBudgetExceeded("interpretation authorized exact reads exhausted")
        return tuple(item for key, item in self._segments.items() if key in requested)

    def payload(self, *, originals=None):
        return self._payload(self._window if originals is None else originals, self.requests[-1:])

    def _payload(self, originals, requests):
        return {
            "inventory": tuple(e.model_dump(mode="json") for e in self.inventory),
            "segments": tuple(p.model_dump(mode="json") for p in originals),
            "read_requests": requests,
        }
