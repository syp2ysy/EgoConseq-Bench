"""Test-only aggregate for focused collection modules.

Production code imports the module that owns each operation.  The larger
characterization suites retain one compact namespace so monkeypatches continue
to describe complete collection scenarios without recreating a production
compatibility facade.
"""

from pipeline import collection_cli as _cli
from pipeline import collection_proposals as _proposals
from pipeline import collection_support as _support
from pipeline import collection_runtime as _runtime
from scripts.collect import main


for _module in (_cli, _proposals, _support, _runtime):
    globals().update({
        _name: _value
        for _name, _value in vars(_module).items()
        if not _name.startswith("__")
    })
