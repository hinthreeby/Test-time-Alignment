from abc import ABC, abstractmethod


class SignalProvider(ABC):
    name = "base"

    @abstractmethod
    def score(self, prefix_ids, candidate_ids):
        pass
