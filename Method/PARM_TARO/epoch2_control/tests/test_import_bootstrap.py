import os,unittest
from pathlib import Path
from PARM_TARO.adaptive_parm.token_headroom.path_bootstrap import required_python_paths,subprocess_environment

class ImportBootstrapTests(unittest.TestCase):
    def test_order_and_existing_pythonpath_preserved(self):
        root=Path(__file__).resolve().parents[4];paths=required_python_paths(root)
        self.assertEqual(paths[0],root/"Method/Router_Apdative")
        env=subprocess_environment(root,{"PYTHONPATH":"/existing/a:/existing/b"})
        values=env["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(values[:3],[str(path) for path in paths]);self.assertEqual(values[3:],["/existing/a","/existing/b"])
