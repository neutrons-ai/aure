"""Restricted evaluator for parameter expressions in a ``ModelDefinition``.

Derived parameters (``ModelDefinition["derived_parameters"]``) reparametrize a
model by making a *combination* of raw parameters free and deriving one raw
parameter from it — e.g. fitting the surface excess
``Gamma = (SEI.rho - dTHF.rho) * SEI.thickness`` and letting ``SEI.rho`` follow.
Both the assignment and its physicality guards are written as expression
strings, so something has to turn a string into a ``bumps`` parameter
expression.

That something is emphatically **not** :func:`eval`. These strings can be
written by an LLM, and they arrive through the same channels as any other model
JSON. This module walks the parsed AST and admits only what an algebraic
reparametrization needs: numbers, dotted parameter names, unary minus, the five
arithmetic operators, and (for guards) the four ordering comparisons. Anything
else — a call, a subscript, an attribute on something that is not a parameter
name, a walrus — raises :class:`ExpressionError` naming what was rejected.

Names are resolved against a namespace supplied by the caller (the builder,
which knows the sample). Two spellings of an SLD are accepted because the
codebase already uses both: ``SEI.rho`` (how a scientist writes it) and
``SEI.material.rho`` (refl1d's attribute path, and what the cross-state tie
specs use). They canonicalize to the same parameter.
"""

from __future__ import annotations

import ast
from typing import Any, Mapping


class ExpressionError(ValueError):
    """Raised when an expression is malformed, unsafe, or names something unknown."""


# Operators an algebraic reparametrization can legitimately need. Modulo,
# floor-division and the bitwise operators are excluded deliberately: they have
# no meaning on a fit parameter, and bumps does not overload them.
_BINOPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "**",
}

_COMPARES = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}

#: ``X.rho`` is accepted as a synonym for refl1d's ``X.material.rho``.
_ATTR_ALIASES = {"rho": "material.rho", "sld": "material.rho"}


def _dotted_name(node: ast.AST) -> str | None:
    """Flatten ``Attribute``/``Name`` chains to ``"a.b.c"``; None if not a name.

    ``SEI.material.rho`` parses as nested ``Attribute`` nodes, so the chain has
    to be walked back to the root ``Name`` to recover the spelling the user
    actually wrote.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def canonical_name(name: str) -> str:
    """Canonicalize a dotted parameter name (``X.rho`` -> ``X.material.rho``)."""
    layer, _, attr = name.partition(".")
    if not attr:
        return name
    return f"{layer}.{_ATTR_ALIASES.get(attr, attr)}"


def _resolve(name: str, namespace: Mapping[str, Any]) -> Any:
    for candidate in (name, canonical_name(name)):
        if candidate in namespace:
            return namespace[candidate]
    known = sorted(namespace)
    raise ExpressionError(
        f"unknown parameter {name!r}; known names: {', '.join(known) or '(none)'}"
    )


def _eval_node(node: ast.AST, namespace: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(
                f"only numeric constants are allowed, got {node.value!r}"
            )
        return node.value
    if isinstance(node, (ast.Name, ast.Attribute)):
        name = _dotted_name(node)
        if name is None:
            raise ExpressionError("attribute access is only allowed on parameter names")
        return _resolve(name, namespace)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval_node(node.operand, namespace)
        if isinstance(node.op, ast.UAdd):
            return _eval_node(node.operand, namespace)
        raise ExpressionError(f"unsupported unary operator {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _BINOPS:
            raise ExpressionError(
                f"unsupported operator {_op_name(op)}; allowed: "
                + " ".join(_BINOPS.values())
            )
        left = _eval_node(node.left, namespace)
        right = _eval_node(node.right, namespace)
        if op is ast.Add:
            return left + right
        if op is ast.Sub:
            return left - right
        if op is ast.Mult:
            return left * right
        if op is ast.Div:
            return left / right
        return left**right
    raise ExpressionError(f"unsupported syntax: {type(node).__name__}")


def _op_name(op: type) -> str:
    return {ast.Mod: "%", ast.FloorDiv: "//"}.get(op, op.__name__)


def _parse(expr: str) -> ast.AST:
    if not isinstance(expr, str) or not expr.strip():
        raise ExpressionError("expression must be a non-empty string")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"could not parse {expr!r}: {exc.msg}") from exc
    return tree.body


def evaluate(expr: str, namespace: Mapping[str, Any]) -> Any:
    """Evaluate an arithmetic expression over *namespace*.

    Returns whatever the operands produce — a ``bumps`` parameter expression
    when any operand is a parameter, a plain float when they are all constants.
    """
    return _eval_node(_parse(expr), namespace)


def evaluate_constraint(expr: str, namespace: Mapping[str, Any]) -> Any:
    """Evaluate a single ordering comparison into a ``bumps`` ``Constraint``.

    Only one comparison per expression: ``a < b < c`` chains to two constraints
    in Python's grammar but to an ambiguous object in bumps, so it is rejected
    with a message telling the caller to write the two guards separately.
    """
    node = _parse(expr)
    if not isinstance(node, ast.Compare):
        raise ExpressionError(
            f"{expr!r} is not a comparison; a physicality guard looks like "
            '"SEI.rho > 0"'
        )
    if len(node.ops) != 1:
        raise ExpressionError(
            f"{expr!r} chains comparisons; write each guard as its own entry"
        )
    op = type(node.ops[0])
    if op not in _COMPARES:
        raise ExpressionError(
            f"unsupported comparison in {expr!r}; allowed: "
            + " ".join(_COMPARES.values())
        )
    left = _eval_node(node.left, namespace)
    right = _eval_node(node.comparators[0], namespace)
    if op is ast.Lt:
        return left < right
    if op is ast.LtE:
        return left <= right
    if op is ast.Gt:
        return left > right
    return left >= right


def referenced_names(expr: str) -> set[str]:
    """Every dotted name an expression mentions, canonicalized.

    Used to validate a declaration against the model's structure before a fit
    is ever built, so a typo is a config error rather than a mid-fit crash.
    """
    names: set[str] = set()

    def _walk(node: ast.AST) -> None:
        # A dotted name is recorded whole and NOT descended into: walking
        # `SEI.material.rho` blindly would also yield the prefixes `SEI` and
        # `SEI.material`, which name nothing and would fail validation.
        if isinstance(node, (ast.Name, ast.Attribute)):
            dotted = _dotted_name(node)
            if dotted:
                names.add(canonical_name(dotted))
                return
        for child in ast.iter_child_nodes(node):
            _walk(child)

    _walk(_parse(expr))
    return names
