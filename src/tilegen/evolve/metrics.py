# src/tilegen/evolve/metrics.py
#
# Diagnostic panel for evolution campaigns: the optimization targets and
# diagnostic indicators are reported as a panel (not a single score), and
# persisted next to the knowledge store.

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CampaignMetrics:
    """One campaign's diagnostic panel (submissions, promotions, rollbacks, revisions)."""

    submissions: int = 0
    effective_promotions: int = 0
    rollbacks: int = 0
    m0_revisions: int = 0
    m0_revisions_blocked: int = 0
    noise_rejections: int = 0

    @property
    def promotion_rate(self) -> float:
        return self.effective_promotions / self.submissions if self.submissions else 0.0

    @property
    def rollback_rate(self) -> float:
        return self.rollbacks / self.submissions if self.submissions else 0.0

    @property
    def m0_yield(self) -> float:
        return self.effective_promotions / self.m0_revisions if self.m0_revisions else 0.0

    def panel(self) -> dict:
        return {
            **asdict(self),
            "promotion_rate": round(self.promotion_rate, 4),
            "rollback_rate": round(self.rollback_rate, 4),
            "m0_yield": round(self.m0_yield, 4),
        }

    def record(self, root: Path, name: str = "metrics") -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{name}.json"
        entries = []
        if path.exists():
            try:
                entries = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                entries = []
        entries.append({**self.panel(), "recorded_at": time.time()})
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


__all__ = ["CampaignMetrics"]
