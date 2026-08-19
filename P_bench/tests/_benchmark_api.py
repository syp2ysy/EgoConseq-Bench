"""Test-only aggregate for the benchmark contract and focused builders."""

from pipeline import benchmark as _contract
from pipeline import benchmark_builders as _builders
from pipeline import benchmark_tasks as _tasks


for _module in (_contract, _tasks, _builders):
    globals().update({
        _name: _value
        for _name, _value in vars(_module).items()
        if not _name.startswith("__")
    })
