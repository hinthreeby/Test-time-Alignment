import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from PARM_TARO.epoch2_control.resume_compat import TrustedTrainerStateLoadShim

class ResumeCompatibilityTests(unittest.TestCase):
    def test_scope_and_pytorch_26_default(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/"checkpoint-200";checkpoint.mkdir()
            trusted=checkpoint/"rng_state.pth";arbitrary=checkpoint/"arbitrary.pth"
            payload={"numpy":np.random.get_state()};torch.save(payload,trusted);torch.save(payload,arbitrary)
            real_load=torch.load
            class FakeTorch:pass
            fake=FakeTorch()
            def pytorch26_load(*args,**kwargs):
                kwargs.setdefault("weights_only",True);return real_load(*args,**kwargs)
            fake.load=pytorch26_load
            with self.assertRaises(Exception):fake.load(trusted)
            shim=TrustedTrainerStateLoadShim(checkpoint,fake);shim.install()
            loaded=fake.load(trusted)
            self.assertIn("numpy",loaded);self.assertEqual(shim.events,["rng_state.pth"])
            with self.assertRaises(Exception):fake.load(arbitrary)
            with self.assertRaises(Exception):fake.load(trusted,weights_only=True)
            self.assertEqual(shim.events,["rng_state.pth"])
