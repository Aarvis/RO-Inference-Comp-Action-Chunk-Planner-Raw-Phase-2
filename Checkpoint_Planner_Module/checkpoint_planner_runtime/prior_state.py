from __future__ import annotations


class PlannerPriorState:
    def __init__(self, *, transition_completion_threshold: float, done_index: int) -> None:
        self.transition_completion_threshold = float(transition_completion_threshold)
        self.done_index = int(done_index)
        self.last_completed_checkpoint = 0

    def reset(self) -> None:
        self.last_completed_checkpoint = 0

    def update(
        self,
        *,
        chosen_checkpoint_index: int,
        phase_pred_index: int,
        trans_pred: float,
    ) -> int:
        before = int(self.last_completed_checkpoint)
        if int(chosen_checkpoint_index) >= self.done_index:
            self.last_completed_checkpoint = self.done_index
        elif int(phase_pred_index) == 1 and float(trans_pred) >= self.transition_completion_threshold:
            self.last_completed_checkpoint = max(self.last_completed_checkpoint, int(chosen_checkpoint_index) + 1)
        return before
