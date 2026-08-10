from Method.MultiSignal.signals.base import SignalProvider


class FutureValueSignal(SignalProvider):
    name = "future"

    def __init__(self, cdq_adapter):
        self.adapter = cdq_adapter

    def score(self, prefix_ids, candidate_ids):
        return self.adapter.get_signal_scores(prefix_ids, candidate_ids)
