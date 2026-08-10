from Method.MultiSignal.signals.base import SignalProvider


class PrefixRewardSignal(SignalProvider):
    name = "prefix"

    def __init__(self, rad_adapter):
        self.adapter = rad_adapter

    def score(self, prefix_ids, candidate_ids):
        return self.adapter.get_signal_scores(prefix_ids, candidate_ids)
