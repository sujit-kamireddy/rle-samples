"""A bounded DAX interpreter for the simulated model, never a substring router.

Unsupported constructs return explicit errors. Cells retain source references so
only values actually returned (after filtering/projection/limits) earn evidence
credit. This is not a replacement for the production semantic-model engine.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any

Source = tuple[str, int, str]


class DaxError(ValueError):
    def __init__(self, message: str, code: str = "InvalidDaxQuery") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Literal:
    value: Any


@dataclass(frozen=True)
class Name:
    value: str


@dataclass(frozen=True)
class Column:
    name: str
    table: str | None = None


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class Binary:
    op: str
    left: Expr
    right: Expr


Expr = Literal | Name | Column | Call | Binary
_TOKEN = re.compile(
    r'\s+|//[^\n]*|--[^\n]*|/\*.*?\*/|"(?:[^"]|"")*"|'
    r"'(?:[^']|'')*'|\[(?:[^\]]|\]\])+\]|\d+(?:\.\d+)?|"
    r"[A-Za-z_]\w*|&&|\|\||<=|>=|<>|==|[=<>+\-*/&(),;]",
    re.DOTALL,
)
_PRECEDENCE = {"||": 1, "&&": 2, "=": 3, "==": 3, "<>": 3,
               "<": 3, ">": 3, "<=": 3, ">=": 3, "&": 4,
               "+": 5, "-": 5, "*": 6, "/": 6}
_OPERATORS = {"=": operator.eq, "==": operator.eq, "<>": operator.ne,
              "<": operator.lt, ">": operator.gt, "<=": operator.le,
              ">=": operator.ge, "+": operator.add, "-": operator.sub,
              "*": operator.mul, "/": operator.truediv}
_SCALAR_ARITY = {
    "MAX": (1, 1), "MIN": (1, 1), "SUM": (1, 1), "AVERAGE": (1, 1),
    "FIRSTNONBLANK": (2, 2), "COUNTROWS": (1, 1), "TRUE": (0, 0),
    "FALSE": (0, 0), "BLANK": (0, 0), "IF": (2, 3),
    "CONTAINSSTRING": (2, 2), "AND": (2, 2), "OR": (2, 2),
    "ISBLANK": (1, 1), "NOT": (1, 1), "LOWER": (1, 1), "UPPER": (1, 1),
}
SUPPORTED = (
    "EVALUATE table, FILTER, SELECTCOLUMNS, ADDCOLUMNS, SUMMARIZE, "
    "SUMMARIZECOLUMNS, TOPN, ROW, and ORDER BY; scalar comparisons, "
    "CONTAINSSTRING, MAX/MIN/SUM/AVERAGE, FIRSTNONBLANK and COUNTROWS"
)


class Parser:
    def __init__(self, text: str) -> None:
        if len(text) > 32768:
            raise DaxError("The simulated DAX query exceeds 32768 characters.")
        self.tokens: list[str] = []
        position = 0
        while position < len(text):
            match = _TOKEN.match(text, position)
            if match is None:
                raise DaxError(f"Invalid DAX token near {text[position:position + 40]!r}.")
            token = match.group()
            position = match.end()
            if token.isspace() or token.startswith(("//", "--", "/*")):
                continue
            self.tokens.append(token)
        self.index = 0
        self.depth = 0

    def peek(self) -> str:
        return self.tokens[self.index] if self.index < len(self.tokens) else ""

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if not token or (expected is not None and token.casefold() != expected.casefold()):
            raise DaxError(f"Expected {expected or 'an expression'}, received {token or 'end of query'!r}.")
        self.index += 1
        return token

    def expression(self, precedence: int = 0) -> Expr:
        self.depth += 1
        if self.depth > 64:
            raise DaxError("DAX expressions may nest at most 64 levels.")
        token = self.take()
        if token == "(":
            result = self.expression()
            self.take(")")
        elif token == "-":
            result = Binary("*", Literal(-1), self.expression(7))
        elif token.startswith('"'):
            result = Literal(token[1:-1].replace('""', '"'))
        elif token[0].isdigit():
            if len(token) > 64:
                raise DaxError("Numeric literals may contain at most 64 characters.")
            result = Literal(float(token) if "." in token else int(token))
        elif token.startswith("["):
            result = Column(token[1:-1].replace("]]", "]"))
        elif token.startswith("'") or re.fullmatch(r"[A-Za-z_]\w*", token):
            name = token[1:-1].replace("''", "'") if token.startswith("'") else token
            if self.peek() == "(":
                self.take("(")
                args = []
                if self.peek() != ")":
                    args.append(self.expression())
                    while self.peek() == ",":
                        self.take(",")
                        args.append(self.expression())
                self.take(")")
                result = Call(name.upper(), tuple(args))
            elif self.peek().startswith("["):
                result = Column(self.take()[1:-1].replace("]]", "]"), name)
            elif name.upper() in ("TRUE", "FALSE"):
                result = Literal(name.upper() == "TRUE")
            else:
                result = Name(name)
        else:
            raise DaxError(f"Unexpected DAX token {token!r}.")
        while self.peek() in _PRECEDENCE and _PRECEDENCE[self.peek()] >= precedence:
            op = self.take()
            result = Binary(op, result, self.expression(_PRECEDENCE[op] + 1))
        self.depth -= 1
        return result

    def query(self) -> tuple[Expr, list[tuple[Expr, bool]]]:
        self.take("EVALUATE")
        expression = self.expression()
        ordering = []
        if self.peek().upper() == "ORDER":
            self.take("ORDER")
            self.take("BY")
            while True:
                column = self.expression()
                descending = False
                if self.peek().upper() in ("ASC", "DESC"):
                    descending = self.take().upper() == "DESC"
                ordering.append((column, descending))
                if self.peek() != ",":
                    break
                self.take(",")
        if self.peek() == ";":
            self.take(";")
        if self.peek():
            raise DaxError(f"Unexpected trailing token {self.peek()!r}; SQL WHERE and pipes are not DAX.")
        return expression, ordering


@dataclass(frozen=True)
class Cell:
    value: Any
    sources: frozenset[Source] = frozenset()


@dataclass
class Table:
    name: str | None
    columns: tuple[str, ...]
    rows: list[dict[str, Cell]]


@dataclass
class QueryResult:
    rows: list[dict[str, Any]]
    sources: set[Source]
    total_rows: int


def _arity(call: Call, minimum: int, maximum: int | None = None) -> None:
    maximum = minimum if maximum is None else maximum
    if not minimum <= len(call.args) <= maximum:
        raise DaxError(f"{call.name} expects {minimum}..{maximum} arguments.")


class Evaluator:
    def __init__(self, tables: dict[str, list[dict[str, Any]]],
                 columns: dict[str, tuple[str, ...]]) -> None:
        self.tables = tables
        self.columns = columns

    def _name(self, name: str) -> str:
        for known in self.tables:
            if known.casefold() == name.casefold():
                return known
        raise DaxError(
            f"Unknown table {name!r}. Available tables: {', '.join(self.tables)}. "
            "Call GetSemanticModelSchema before querying."
        )

    def _column(self, column: Column, table: Table) -> str:
        if column.table and self._name(column.table) != table.name:
            raise DaxError("Cross-table expressions are not supported by the simulator.",
                           "UnsupportedDaxQuery")
        if column.table and column.name.casefold() not in {
            name.casefold() for name in self.columns[self._name(column.table)]
        }:
            raise DaxError(f"Unknown model column {column.table}[{column.name}].")
        for name in table.columns:
            if name.casefold() == column.name.casefold():
                return name
        raise DaxError(f"Unknown column [{column.name}]; available: {', '.join(table.columns)}.")

    def validate(self, expression: Expr, table: Table) -> None:
        if isinstance(expression, Column):
            self._column(expression, table)
        elif isinstance(expression, Binary):
            self.validate(expression.left, table)
            self.validate(expression.right, table)
        elif isinstance(expression, Call):
            if expression.name not in _SCALAR_ARITY:
                raise DaxError(f"Unsupported scalar function {expression.name}.",
                               "UnsupportedDaxQuery")
            _arity(expression, *_SCALAR_ARITY[expression.name])
            if expression.name == "COUNTROWS":
                self.table(expression.args[0], table)
                return
            for arg in expression.args:
                self.validate(arg, table)
        elif isinstance(expression, Name):
            raise DaxError("A scalar column reference must use brackets, e.g. [company].")

    def scalar(self, expression: Expr, table: Table,
               row: dict[str, Cell] | None = None) -> Cell:
        if isinstance(expression, Literal):
            return Cell(expression.value)
        if isinstance(expression, Column):
            key = self._column(expression, table)
            cells = [row[key]] if row is not None else [r[key] for r in table.rows]
            if not cells:
                return Cell(None)
            if any(cell.value != cells[0].value for cell in cells):
                raise DaxError(f"A single value for [{key}] cannot be determined; use an aggregate.")
            return Cell(cells[0].value, frozenset().union(*(c.sources for c in cells)))
        if isinstance(expression, Binary):
            left = self.scalar(expression.left, table, row)
            right = self.scalar(expression.right, table, row)
            a, b = left.value, right.value
            if expression.op in ("&&", "||"):
                if not isinstance(a, bool) or not isinstance(b, bool):
                    raise DaxError("Logical DAX operators require Boolean expressions.")
                value = a and b if expression.op == "&&" else a or b
            elif expression.op == "&":
                value = str(a or "") + str(b or "")
            else:
                if isinstance(a, str) and isinstance(b, str) and expression.op in (
                    "=", "==", "<>", "<", ">", "<=", ">="
                ):
                    a, b = a.casefold(), b.casefold()
                try:
                    value = _OPERATORS[expression.op](a, b)
                except (TypeError, ZeroDivisionError) as error:
                    raise DaxError(f"Invalid operands for {expression.op}: {error}") from error
            return Cell(value, left.sources | right.sources)
        if not isinstance(expression, Call):
            raise DaxError("A scalar column reference must use brackets, e.g. [company].")
        call = expression
        if call.name in ("MAX", "MIN", "SUM", "AVERAGE", "FIRSTNONBLANK"):
            _arity(call, 2 if call.name == "FIRSTNONBLANK" else 1)
            self.validate(call.args[0], table)
            values = [self.scalar(call.args[0], table, r) for r in table.rows]
            values = [cell for cell in values if cell.value is not None]
            if call.name == "FIRSTNONBLANK":
                values = [
                    self.scalar(call.args[0], table, r) for r in table.rows
                    if self.scalar(call.args[0], table, r).value is not None
                    and self.scalar(call.args[1], table, r).value is not None
                ]
                return min(
                    values,
                    key=lambda cell: cell.value.casefold()
                    if isinstance(cell.value, str) else cell.value,
                ) if values else Cell(None)
            if not values:
                return Cell(None)
            if call.name in ("MAX", "MIN"):
                chooser = max if call.name == "MAX" else min
                return chooser(values, key=lambda cell: cell.value)
            if any(not isinstance(cell.value, (int, float)) for cell in values):
                raise DaxError(f"{call.name} requires numeric values.")
            total = sum(cell.value for cell in values)
            return Cell(total if call.name == "SUM" else total / len(values),
                        frozenset().union(*(cell.sources for cell in values)))
        if call.name == "COUNTROWS":
            _arity(call, 1)
            return Cell(len(self.table(call.args[0], table).rows))
        if call.name in ("TRUE", "FALSE", "BLANK"):
            _arity(call, 0)
            return Cell(None if call.name == "BLANK" else call.name == "TRUE")
        if call.name == "IF":
            _arity(call, 2, 3)
            condition = self.scalar(call.args[0], table, row).value
            if not isinstance(condition, bool):
                raise DaxError("IF requires a Boolean condition.")
            branch = call.args[1] if condition else (call.args[2] if len(call.args) == 3 else Literal(None))
            return self.scalar(branch, table, row)
        if call.name in ("CONTAINSSTRING", "AND", "OR"):
            _arity(call, 2)
            a, b = (self.scalar(arg, table, row) for arg in call.args)
            if call.name == "CONTAINSSTRING":
                if not isinstance(a.value, str) or not isinstance(b.value, str):
                    raise DaxError("CONTAINSSTRING requires text arguments.")
                if any(char in b.value for char in "*?~"):
                    raise DaxError("CONTAINSSTRING wildcards are not supported by the simulator.",
                                   "UnsupportedDaxQuery")
                return Cell(b.value.casefold() in a.value.casefold())
            if not isinstance(a.value, bool) or not isinstance(b.value, bool):
                raise DaxError(f"{call.name} requires Boolean arguments.")
            return Cell(a.value and b.value if call.name == "AND" else a.value or b.value)
        if call.name in ("ISBLANK", "NOT", "LOWER", "UPPER"):
            _arity(call, 1)
            cell = self.scalar(call.args[0], table, row)
            if call.name == "ISBLANK":
                return Cell(cell.value is None)
            if call.name == "NOT" and isinstance(cell.value, bool):
                return Cell(not cell.value)
            if call.name in ("LOWER", "UPPER") and isinstance(cell.value, str):
                return Cell(cell.value.lower() if call.name == "LOWER" else cell.value.upper())
            raise DaxError(f"Invalid argument type for {call.name}.")
        raise DaxError(f"Unsupported scalar function {call.name}. Supported: {SUPPORTED}.",
                       "UnsupportedDaxQuery")

    def _pairs(self, args: tuple[Expr, ...]) -> list[tuple[str, Expr]]:
        if not args or len(args) % 2:
            raise DaxError('Expected pairs of "output name", expression.')
        pairs = []
        names = set()
        for alias, expression in zip(args[::2], args[1::2], strict=True):
            if not isinstance(alias, Literal) or not isinstance(alias.value, str):
                raise DaxError("An output-column alias must be a quoted string.")
            if alias.value.casefold() in names:
                raise DaxError(f"Duplicate output-column alias {alias.value!r}.")
            names.add(alias.value.casefold())
            pairs.append((alias.value, expression))
        return pairs

    def ordered(self, table: Table, ordering: list[tuple[Expr, bool]]) -> Table:
        for expression, _ in ordering:
            self.validate(expression, table)

        def compare(left: dict[str, Cell], right: dict[str, Cell]) -> int:
            for expression, descending in ordering:
                a, b = (self.scalar(expression, table, row).value for row in (left, right))
                if a == b:
                    continue
                if a is None or b is None:
                    result = -1 if a is None else 1
                else:
                    if isinstance(a, str) and isinstance(b, str):
                        a, b = a.casefold(), b.casefold()
                    result = (a > b) - (a < b)
                if result:
                    return -result if descending else result
            return 0

        return Table(table.name, table.columns, sorted(table.rows, key=cmp_to_key(compare)))

    def table(self, expression: Expr, context: Table | None = None) -> Table:
        if isinstance(expression, Name):
            name = self._name(expression.value)
            if context is not None and context.name == name:
                return context
            rows = [
                {key: Cell(value, frozenset({(name, index, key)})) for key, value in row.items()}
                for index, row in enumerate(self.tables[name])
            ]
            return Table(name, self.columns[name], rows)
        if not isinstance(expression, Call):
            raise DaxError("EVALUATE requires a table expression.")
        call = expression
        if call.name == "ROW":
            pairs = self._pairs(call.args)
            empty = Table(None, (), [])
            return Table(None, tuple(name for name, _ in pairs), [
                {name: self.scalar(value, context or empty) for name, value in pairs}
            ])
        if call.name == "SUMMARIZECOLUMNS":
            columns = []
            filters = []
            index = 0
            while index < len(call.args) and not isinstance(call.args[index], Literal):
                arg = call.args[index]
                if isinstance(arg, Column):
                    columns.append(arg)
                elif isinstance(arg, Call) and arg.name == "FILTER":
                    filters.append(arg)
                else:
                    raise DaxError("Unsupported SUMMARIZECOLUMNS argument.", "UnsupportedDaxQuery")
                index += 1
            if not columns or columns[0].table is None:
                raise DaxError("SUMMARIZECOLUMNS requires a qualified grouping column.",
                               "UnsupportedDaxQuery")
            base: Expr = Name(columns[0].table)
            for filter_call in filters:
                _arity(filter_call, 2)
                if filter_call.args[0] != base:
                    raise DaxError("SUMMARIZECOLUMNS filters must use the grouping table.",
                                   "UnsupportedDaxQuery")
            table = self.table(base, context)
            for filter_call in filters:
                table = self.table(filter_call, table)
            return self._summarize(table, tuple(columns) + call.args[index:])
        if call.name not in ("FILTER", "SELECTCOLUMNS", "ADDCOLUMNS", "SUMMARIZE",
                             "TOPN", "ALL", "DISTINCT"):
            raise DaxError(f"Unsupported table function {call.name}. Supported: {SUPPORTED}.",
                           "UnsupportedDaxQuery")
        _arity(call, 1, 100)
        if call.name == "TOPN":
            _arity(call, 3, 99)
            count = self.scalar(call.args[0], context or Table(None, (), [])).value
            if type(count) is not int or count < 0:
                raise DaxError("TOPN requires a non-negative integer row count.")
            table = self.table(call.args[1], context)
            ordering = []
            index = 2
            while index < len(call.args):
                key = call.args[index]
                index += 1
                descending = True
                if index < len(call.args):
                    direction = call.args[index]
                    if isinstance(direction, Name) and direction.value.upper() in ("ASC", "DESC"):
                        descending = direction.value.upper() == "DESC"
                        index += 1
                    elif isinstance(direction, Literal) and direction.value in (0, 1):
                        descending = direction.value == 0
                        index += 1
                ordering.append((key, descending))
            ordered = self.ordered(table, ordering)
            end = min(count, len(ordered.rows))
            if end:
                def keys(row):
                    return tuple(self.scalar(key, table, row).value for key, _ in ordering)
                while end < len(ordered.rows) and keys(ordered.rows[end]) == keys(ordered.rows[end - 1]):
                    end += 1
            return Table(table.name, table.columns, ordered.rows[:end])
        table = self.table(call.args[0], context)
        if call.name == "FILTER":
            _arity(call, 2)
            self.validate(call.args[1], table)
            rows = []
            for row in table.rows:
                keep = self.scalar(call.args[1], table, row).value
                if not isinstance(keep, bool):
                    raise DaxError("FILTER requires a Boolean predicate.")
                if keep:
                    rows.append(row)
            return Table(table.name, table.columns, rows)
        if call.name in ("SELECTCOLUMNS", "ADDCOLUMNS"):
            pairs = self._pairs(call.args[1:])
            for _, expression in pairs:
                self.validate(expression, table)
            names = tuple(name for name, _ in pairs)
            if call.name == "ADDCOLUMNS":
                if {name.casefold() for name in names} & {name.casefold() for name in table.columns}:
                    raise DaxError("ADDCOLUMNS cannot replace an existing column.")
                names = table.columns + names
            rows = []
            for row in table.rows:
                projected = dict(row) if call.name == "ADDCOLUMNS" else {}
                projected.update({name: self.scalar(expr, table, row) for name, expr in pairs})
                rows.append(projected)
            return Table(table.name, names, rows)
        if call.name == "SUMMARIZE":
            return self._summarize(table, call.args[1:])
        _arity(call, 1)
        if call.name == "ALL":
            if not isinstance(call.args[0], Name):
                raise DaxError("ALL requires a base table.", "UnsupportedDaxQuery")
            return self.table(call.args[0])
        rows = []
        seen = set()
        for row in table.rows:
            key = tuple(row[name].value for name in table.columns)
            if key not in seen:
                seen.add(key)
                rows.append(row)
        return Table(table.name, table.columns, rows)

    def _summarize(self, table: Table, args: tuple[Expr, ...]) -> Table:
        index = 0
        columns = []
        while index < len(args) and isinstance(args[index], Column):
            columns.append(self._column(args[index], table))
            index += 1
        pairs = self._pairs(args[index:]) if index < len(args) else []
        if not columns and not pairs:
            raise DaxError("SUMMARIZE requires grouping columns or aggregates.")
        for _, expression in pairs:
            self.validate(expression, table)
        names = tuple(columns) + tuple(name for name, _ in pairs)
        if len({name.casefold() for name in names}) != len(names):
            raise DaxError("SUMMARIZE output columns must have distinct names.")
        groups: dict[tuple, list[dict[str, Cell]]] = {}
        for row in table.rows:
            groups.setdefault(tuple(row[name].value for name in columns), []).append(row)
        rows = []
        for members in groups.values():
            group = Table(table.name, table.columns, members)
            row = {name: members[0][name] for name in columns}
            row.update({name: self.scalar(value, group) for name, value in pairs})
            rows.append(row)
        return Table(table.name, names, rows)


def evaluate_query(text: str, tables: dict[str, list[dict[str, Any]]],
                   columns: dict[str, tuple[str, ...]], max_rows: int = 250) -> QueryResult:
    if type(max_rows) is not int or max_rows < 1:
        raise DaxError("max_rows must be a positive integer.")
    expression, ordering = Parser(text).query()
    evaluator = Evaluator(tables, columns)
    table = evaluator.table(expression)
    if ordering:
        table = evaluator.ordered(table, ordering)
    returned = table.rows[:max_rows]
    sources = {
        source for row in returned for cell in row.values() for source in cell.sources
        if tables[source[0]][source[1]][source[2]] == cell.value
    }
    return QueryResult(
        rows=[{name: cell.value for name, cell in row.items()} for row in returned],
        sources=sources,
        total_rows=len(table.rows),
    )
