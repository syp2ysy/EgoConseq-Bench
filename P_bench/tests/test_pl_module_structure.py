"""Structural guardrails that must not compete with GT correctness."""

from __future__ import annotations

import ast
import builtins
import collections
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS = (ROOT / "pipeline", ROOT / "scripts")
# These are upper bounds against renewed god modules, not LOC-reduction goals.
# Oracle validation is intentionally explicit; do not split it merely to satisfy
# a cosmetic threshold while the GT trust boundary is still being hardened.
MAX_FUNCTION_LINES = 300
# Oracle-v7 adds explicit, independently testable GT rederivation. Keep the
# guardrail close to the current structure without forcing trust-boundary code
# into compressed or artificial modules merely to preserve the older budget.
MAX_MODULE_LINES = 1_600
# Deliberately absent: aggregate pipeline file-count and line-count budgets.
# They were ratcheted upward six times (24,754 -> 25,300 -> 27,000 -> 27,250 ->
# 27,400 -> 27,600), each time to whatever the tree had just grown to, so they
# never constrained growth. What they did constrain was correction: splitting a
# module by responsibility adds files, so the caps made the right refactor fail
# the gate. Per-module and per-function limits target the actual failure mode.
THIN_CLI_MAX_LINES = 100
THIN_CLIS = (
    ROOT / "scripts" / "collect.py",
)


def _production_modules():
    for root in PRODUCTION_ROOTS:
        yield from sorted(root.glob("*.py"))


def test_production_functions_stay_below_the_refactor_limit():
    oversized = []
    for path in _production_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            size = node.end_lineno - node.lineno + 1
            if size > MAX_FUNCTION_LINES:
                oversized.append(f"{path.relative_to(ROOT)}:{node.lineno} "
                                 f"{node.name} ({size})")
    assert oversized == []


def test_production_modules_stay_below_the_refactor_limit():
    oversized = {
        str(path.relative_to(ROOT)): len(path.read_text().splitlines())
        for path in _production_modules()
        if len(path.read_text().splitlines()) > MAX_MODULE_LINES
    }
    assert oversized == {}


def test_primary_cli_entrypoints_remain_thin():
    oversized = {
        str(path.relative_to(ROOT)): len(path.read_text().splitlines())
        for path in THIN_CLIS
        if len(path.read_text().splitlines()) > THIN_CLI_MAX_LINES
    }
    assert oversized == {}


def _pipeline_import_graph():
    modules = {
        ".".join(path.relative_to(ROOT).with_suffix("").parts): path
        for path in (ROOT / "pipeline").glob("*.py")
    }
    graph = collections.defaultdict(set)
    for module, path in modules.items():
        tree = ast.parse(path.read_text())
        # A function-local import may avoid an initialization crash, but it
        # still creates an architectural dependency. The refactor gate checks
        # the complete dependency graph rather than permitting hidden cycles.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "pipeline":
                graph[module].update(
                    f"pipeline.{alias.name}"
                    for alias in node.names
                    if f"pipeline.{alias.name}" in modules
                )
            elif (isinstance(node, ast.ImportFrom) and node.module in modules):
                graph[module].add(node.module)
            elif isinstance(node, ast.Import):
                graph[module].update(
                    alias.name for alias in node.names if alias.name in modules
                )
    return modules, graph


def _cyclic_pipeline_components():
    modules, graph = _pipeline_import_graph()
    index = 0
    indices = {}
    lowlinks = {}
    stack = []
    on_stack = set()
    cyclic = []

    def visit(module):
        nonlocal index
        indices[module] = lowlinks[module] = index
        index += 1
        stack.append(module)
        on_stack.add(module)
        for dependency in graph[module]:
            if dependency not in indices:
                visit(dependency)
                lowlinks[module] = min(
                    lowlinks[module], lowlinks[dependency])
            elif dependency in on_stack:
                lowlinks[module] = min(
                    lowlinks[module], indices[dependency])
        if lowlinks[module] != indices[module]:
            return
        component = []
        while True:
            dependency = stack.pop()
            on_stack.remove(dependency)
            component.append(dependency)
            if dependency == module:
                break
        if len(component) > 1:
            cyclic.append(tuple(sorted(component)))

    for module in sorted(modules):
        if module not in indices:
            visit(module)
    return sorted(cyclic)


def test_pipeline_modules_have_no_import_cycles_or_dynamic_facades():
    assert _cyclic_pipeline_components() == []
    assert not (ROOT / "pipeline" / "module_facade.py").exists()


def test_production_modules_do_not_import_private_pipeline_apis():
    violations = []
    for path in _production_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.startswith("pipeline.")):
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    violations.append(
                        f"{path.name}:{node.lineno} "
                        f"{node.module}.{alias.name}")
    assert violations == []


_BUILTIN_NAMES = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__spec__", "__package__",
}


def _names_bound_anywhere(tree):
    """Over-approximate every name the module can bind.

    Deliberately ignores scope: a name bound in any function counts as bound
    for the whole module.  The approximation is one-sided, so a name outside
    this set cannot be resolved by local, enclosing, global, or builtin
    lookup, and flagging it is never a false positive.
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Lambda)):
            if not isinstance(node, ast.Lambda):
                names.add(node.name)
            arguments = getattr(node, "args", None)
            if arguments is not None:
                for argument in (
                        list(arguments.args)
                        + list(getattr(arguments, "posonlyargs", []))
                        + list(arguments.kwonlyargs)
                        + [arguments.vararg, arguments.kwarg]):
                    if argument is not None:
                        names.add(argument.arg)
        if isinstance(node, ast.Name) and isinstance(
                node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                alias.asname or alias.name.split(".")[0]
                for alias in node.names)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def test_production_modules_reference_no_undefined_names():
    """Catch cross-module wiring that only fails on an untaken branch.

    The test suites share ``tests/_collection_api``, which merges several
    pipeline namespaces into one object.  A helper called from module A but
    defined only in module B therefore resolves in tests while raising
    NameError in production, and it does so only once the owning collection
    mode actually runs.  This gate reads production modules directly.
    """
    undefined = []
    for path in _production_modules():
        tree = ast.parse(path.read_text())
        assert not [
            alias for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names if alias.name == "*"
        ], f"{path.relative_to(ROOT)} uses a star import"
        resolvable = _names_bound_anywhere(tree) | _BUILTIN_NAMES
        undefined.extend(
            f"{path.relative_to(ROOT)}:{node.lineno} {node.id}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in resolvable
        )
    assert undefined == []


def test_release_gates_have_one_authoritative_config_source():
    duplicated = set()
    for path in (ROOT / "pipeline").glob("*.py"):
        if path.name == "config.py":
            continue
        tree = ast.parse(path.read_text())
        duplicated.update(
            f"{path.name}:{target.id}"
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id.startswith("RELEASE_")
        )
    assert duplicated == set()


def test_horizontal_direction_has_one_authoritative_implementation():
    definitions = []
    for path in (ROOT / "pipeline").glob("*.py"):
        tree = ast.parse(path.read_text())
        definitions.extend(
            f"{path.name}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name in {"horizontal_direction", "_horizontal_direction"}
        )
    assert len(definitions) == 1
    assert definitions[0].startswith("actions.py:")


def test_quantile_has_one_authoritative_implementation():
    # Anchored on the live authority rather than on whichever module last
    # reimplemented it: naming a suspect module makes the gate die with that
    # module instead of following the contract it is meant to protect.
    # A caller that merely aggregates quantiles is fine; what must stay unique
    # is the interpolation arithmetic, so delegation to the authority clears a
    # helper regardless of its name.
    authority = "io_utils.py"
    authoritative = []
    reimplementations = []
    for path in _production_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if "quantile" not in node.name.lower():
                continue
            label = f"{path.name}:{node.lineno} {node.name}"
            if path.name == authority:
                authoritative.append(label)
                continue
            if not any(
                    isinstance(call, ast.Call) and (
                        (isinstance(call.func, ast.Attribute)
                         and call.func.attr == "linear_quantile")
                        or (isinstance(call.func, ast.Name)
                            and call.func.id == "linear_quantile"))
                    for call in ast.walk(node)):
                reimplementations.append(label)
    assert authoritative
    assert reimplementations == []



def test_git_revision_state_has_one_authoritative_implementation():
    implementations = []
    for path in _production_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                    isinstance(function, ast.Attribute)
                    and function.attr == "run"
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "subprocess"):
                continue
            if any(
                    isinstance(argument, ast.List)
                    and any(
                        isinstance(value, ast.Constant)
                        and value.value == "rev-parse"
                        for value in argument.elts)
                    for argument in node.args):
                implementations.append(str(path.relative_to(ROOT)))
    assert implementations == ["pipeline/io_utils.py"]
