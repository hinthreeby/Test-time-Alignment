from Method.MultiSignal.signals.base import SignalProvider


class TokenRewardSignal(SignalProvider):
    name = "token"

    def __init__(self, genarm_adapter):
        self.adapter = genarm_adapter

    def score(self, prefix_ids, candidate_ids):
        return self.adapter.get_signal_scores(prefix_ids, candidate_ids)
