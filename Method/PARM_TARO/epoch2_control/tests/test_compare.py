import unittest
from pathlib import Path
from PARM_TARO.epoch2_control.compare import metrics

class CompareTests(unittest.TestCase):
    def test_epoch1_metrics_are_readable(self):
        root=Path(__file__).resolve().parents[4]
        result=metrics(root/"results/parm_taro/pareto_scalarization_probe/01_dense_alpha")
        self.assertEqual(len(result),8)
