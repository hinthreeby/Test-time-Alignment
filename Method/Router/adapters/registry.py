from Method.Router.adapters.rad import RADAdapter
from Method.Router.adapters.args import ARGSAdapter
from Method.Router.adapters.cd import CDAdapter
from Method.Router.adapters.genarm import GenARMAdapter


ADAPTERS = {
    "rad": RADAdapter,
    "args": ARGSAdapter,
    "cd-fudge": CDAdapter,
    "cdq": CDAdapter,
    "genarm": GenARMAdapter,
}


def get_adapter(name):
    name = name.lower()
    if name not in ADAPTERS:
        raise ValueError(f"Unknown method: {name}. Available: {list(ADAPTERS)}")
    return ADAPTERS[name]
