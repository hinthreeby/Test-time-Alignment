from Method.MultiSignal.signals.base import SignalProvider


class GlobalRewardSignal(SignalProvider):
    name = "global"

    def __init__(self, args_adapter):
        self.adapter = args_adapter

    def score(self, prefix_ids, candidate_ids):
        return self.adapter.get_signal_scores(prefix_ids, candidate_ids)
