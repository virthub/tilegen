# src/tilegen/evolve/attempt_history.py
#
# Epoch-scoped attempt history for the inner loop: dead-end memory that
# records which decision fields failed under the current structural
# topology. The history resets when the curriculum's m0_epoch advances (a
# structural revision changes which topology the levers were tuned against),
# so a lever that failed under a previous topology is not excluded forever.

from __future__ import annotations

from typing import Optional


class AttemptHistory:
    """Per-epoch dead-end memory, capped (oldest dropped first)."""

    def __init__(self, cap: int = 10):
        self.cap = cap
        self.epoch = 0
        self._entries: list[dict] = []

    def record(
        self,
        fields: list[str],
        failure_mode: str,
        epoch: Optional[int] = None,
    ) -> None:
        epoch = self.epoch if epoch is None else epoch
        if epoch != self.epoch:
            return # stale epoch: never recorded against the current one
        for field in fields:
            self._entries.append({
                "field": field,
                "failure_mode": failure_mode,
                "epoch": epoch,
            })
        while len(self._entries) > self.cap:
            self._entries.pop(0)

    def advance_epoch(self) -> None:
        """A structural revision: open a fresh queue for the new epoch."""
        self.epoch += 1
        self._entries = [e for e in self._entries if e["epoch"] == self.epoch]

    def failed_fields(self, epoch: Optional[int] = None) -> set[str]:
        epoch = self.epoch if epoch is None else epoch
        return {e["field"] for e in self._entries if e["epoch"] == epoch}

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["AttemptHistory"]
