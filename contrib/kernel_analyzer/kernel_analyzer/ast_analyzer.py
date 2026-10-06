# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AST analyzer for kernel functions."""

import ast
import dataclasses
import logging
import re
from typing import Dict, List, Optional, Tuple


@dataclasses.dataclass
class Issue:
  node: ast.AST
  kernel: str


@dataclasses.dataclass
class DefaultForbidden(Issue):
  def __str__(self):
    return f'"{self.node.arg}" default not allowed.'


@dataclasses.dataclass
class VarArgsForbidden(Issue):
  def __str__(self):
    return f'"{self.node.arg}" varargs not allowed.'


@dataclasses.dataclass
class KwArgsForbidden(Issue):
  def __str__(self):
    return f'"{self.node.arg}" kwargs not allowed.'


@dataclasses.dataclass
class MissingType(Issue):
  def __str__(self):
    return f'"{self.node.arg}" missing type annotation.'


@dataclasses.dataclass
class TypeMismatch(Issue):
  expected_type: str
  source: Optional[str]

  def __str__(self):
    ret = f'"{self.node.arg}" type mismatch, expected {self.expected_type}'
    ret += "." if self.source is None else f" from {self.source}."
    return ret


@dataclasses.dataclass
class InvalidSuffix(Issue):
  source: Optional[str]

  def __str__(self):
    if self.source == "Model":
      return f'"{self.node.arg}" invalid suffix, _in/_out not allowed on Model parameters.'
    elif self.source == "Data":
      return f'"{self.node.arg}" invalid suffix, _in/_out required on Data parameters.'
    else:
      return f'"{self.node.arg}" invalid suffix, _in/_out required.'


@dataclasses.dataclass
class InvalidParamOrder(Issue):
  expected: List[str]
  expected_full: List[str]  # with comments and types
  # function arg range ((start_lineno, start_col), (end_lineno, end_col))
  arg_range: Tuple[Tuple[int, int], Tuple[int, int]]

  def __str__(self):
    return f'"{self.kernel}" invalid parameter order, expected: {", ".join(self.expected)}'


@dataclasses.dataclass
class MissingComment(Issue):
  expected_comment: str

  def __str__(self):
    return f'"{self.node.arg}" should be preceded by comment: {self.expected_comment}'


@dataclasses.dataclass
class InvalidWrite(Issue):
  def __str__(self):
    return f'"{self.node.id}" invalid write: parameter is read-only'


@dataclasses.dataclass
class MissingModuleUnique(Issue):
  def __str__(self):
    return f'"{self.kernel}" nested kernel missing module="unique"'


@dataclasses.dataclass
class SharedModuleCustomBlockDim(Issue):
  def __str__(self):
    return f'"{self.kernel}" is launched with custom block_dim but shares a module. Use @wp.kernel(module="unique").'


@dataclasses.dataclass
class ParenthesizedArraySyntax(Issue):
  def __str__(self):
    return f'"{self.node.arg}" uses parenthesized wp.array(dtype=X) syntax, use wp.array[X] instead'


@dataclasses.dataclass
class MissingBatchModulo(Issue):
  param_name: str

  def __str__(self):
    return f'"{self.param_name}" is a *-dimensioned field; first index must use "% {self.param_name}.shape[0]"'


@dataclasses.dataclass
class DirectWpArrayForbidden(Issue):
  attr_name: str
  class_name: str

  def __str__(self):
    return f'"{self.attr_name}" in class "{self.class_name}" must use array(...) annotation instead of direct wp.array.'


@dataclasses.dataclass
class AmbiguousPrecedence(Issue):
  description: str

  def __str__(self):
    return f"ambiguous operator precedence: {self.description}"


@dataclasses.dataclass
class BitwiseInversionInBoolean(Issue):
  def __str__(self):
    return 'bitwise NOT (~) used as boolean condition; use "not (...)", "!= 0", or explicit flag checks'


@dataclasses.dataclass
class WarpCallInStatic(Issue):
  def __str__(self):
    return f'"wp.{self.node.func.attr}" not allowed inside wp.static(); use pure Python or math.*'


@dataclasses.dataclass
class MissingDeterministicFactory(Issue):
  def __str__(self):
    return f'"{self.kernel}" contains atomic operations but is not enclosed in a kernel factory with a "deterministic" argument'


@dataclasses.dataclass
class MissingDeterministicMaxRecords(Issue):
  reason: str = 'deterministic factory must set "deterministic_max_records" in module_options'

  def __str__(self):
    return f'"{self.kernel}" {self.reason}'


@dataclasses.dataclass
class AtomicTargetConflict(Issue):
  reason: str

  def __str__(self):
    return f'"{self.kernel}" {self.reason}'


# TODO(team): add argument order analyzer.
# this one is tricky because just verifying order does not tell you if the arguments
# match the parameter signature.


def _get_classes(src: str) -> Dict[str, List[Tuple[str, str]]]:
  """Return classes and fields and annotations."""
  ret = {}
  class_name = None
  in_docstring = False
  field_pattern = re.compile(r"^\s+(\w+)\s*:\s*(.*?)(?:\s*#.*)?$")

  for line in src.splitlines():
    if line.startswith("class "):
      class_name = line[len("class ") : -1]
    if line.strip().startswith('"""'):
      in_docstring = True
    if line.strip().endswith('"""'):
      in_docstring = False

    m = field_pattern.match(line)
    if not class_name or not m or in_docstring:
      continue

    ret.setdefault(class_name, []).append((m.group(1), m.group(2)))

  return ret


def _get_function_arg_range(node: ast.FunctionDef, source: str) -> tuple[tuple[int, int], tuple[int, int]]:
  """Gets the function arg range."""
  start_line = node.lineno - 1
  start_char = node.col_offset + len(node.name)
  lines = source.splitlines()
  open_paren = lines[start_line].find("(", start_char)
  start_pos = (start_line, open_paren + 1)

  balance = 1
  curr_line, curr_char = start_line, open_paren + 1
  while balance > 0 and curr_line < len(lines):
    search_line = lines[curr_line]
    while curr_char < len(search_line):
      if search_line[curr_char] == "(":
        balance += 1
      elif search_line[curr_char] == ")":
        balance -= 1
        if balance == 0:
          return start_pos, (curr_line, curr_char)
      curr_char += 1
    curr_line += 1
    curr_char = 0
  return start_pos, (node.end_lineno - 1, node.end_col_offset)


def _get_arg_expected_comment(param_source: str, param_out: bool) -> str:
  if param_source == "Model":
    expected_comment = "# Model:"
  elif param_source == "Data":
    expected_comment = f"# Data {'out' if param_out else 'in'}:"
  else:
    expected_comment = f"# {'Out' if param_out else 'In'}:"
  return expected_comment


def _is_parenthesized(source_lines: List[str], node: ast.AST) -> bool:
  """Check if an AST node is immediately enclosed in parentheses."""
  # Scan backward for '('
  start_l, start_c = node.lineno - 1, node.col_offset - 1
  has_open = False
  while start_l >= 0:
    line = source_lines[start_l]
    while start_c >= 0:
      ch = line[start_c]
      if not ch.isspace() and ch != "\\":
        has_open = ch == "("
        break
      start_c -= 1
    if has_open or (start_c >= 0 and line[start_c] not in (" ", "\t", "\\")):
      break
    start_l -= 1
    if start_l >= 0:
      start_c = len(source_lines[start_l]) - 1

  # Scan forward for ')'
  end_l, end_c = node.end_lineno - 1, node.end_col_offset
  has_close = False
  while end_l < len(source_lines):
    line = source_lines[end_l]
    while end_c < len(line):
      ch = line[end_c]
      if not ch.isspace() and ch != "\\":
        has_close = ch == ")"
        break
      end_c += 1
    if has_close or (end_c < len(line) and line[end_c] not in (" ", "\t", "\\")):
      break
    end_l += 1
    end_c = 0

  return has_open and has_close


def _is_wp_call(node: ast.AST) -> bool:
  return (
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and isinstance(node.func.value, ast.Name)
    and node.func.value.id == "wp"
  )


def analyze(
  source: str,
  filename: str,
  type_source: str,
  check_atomic: bool = False,
) -> List[Issue]:
  """Parses Python code and finds functions with unsorted simple parameters."""
  logging.info(f"Analyzing {filename}...")

  # get class fields and types for Model and Data
  type_classes = _get_classes(type_source)
  field_info = {}
  star_fields = set()  # fields with "*" first dimension

  for field, typ in type_classes.get("Model", []):
    if field == "opt":
      for sfield, styp in type_classes.get("Option", []):
        full_name = field + "_" + sfield
        field_info[full_name] = ("Model", styp, len(field_info))
        if styp.startswith('array("*"'):
          star_fields.add(full_name)
    elif field == "stat":
      for sfield, styp in type_classes.get("Statistic", []):
        full_name = field + "_" + sfield
        field_info[full_name] = ("Model", styp, len(field_info))
        if styp.startswith('array("*"'):
          star_fields.add(full_name)
    else:
      field_info[field] = ("Model", typ, len(field_info))
      if typ.startswith('array("*"'):
        star_fields.add(field)

  for field, typ in type_classes.get("Data", []):
    if field == "efc":
      for sfield, styp in type_classes["Constraint"]:
        field_info[field + "_" + sfield] = ("Data", styp, len(field_info))
    elif field == "contact":
      for sfield, styp in type_classes["Contact"]:
        field_info[field + "_" + sfield] = ("Data", styp, len(field_info))
    else:
      field_info[field] = ("Data", typ, len(field_info))

  try:
    tree = ast.parse(source, filename=filename)
  except SyntaxError as e:
    logging.error(f"Syntax error in {filename}:{e.lineno}: {e.msg}")
    return []

  issues: List[Issue] = []
  for cls_name in ("Model", "Data", "Option", "Statistic", "Constraint"):
    if cls_name in type_classes:
      for attr_name, attr_type in type_classes[cls_name]:
        if attr_type.startswith("wp.array"):
          dummy_node = ast.AST()
          dummy_node.lineno = 1
          dummy_node.end_lineno = 1
          issues.append(DirectWpArrayForbidden(dummy_node, "", attr_name, cls_name))
  source_lines = source.splitlines()

  def _is_kernel(name: str) -> bool:
    """Check if decorator is wp.kernel."""
    return name and (name == "wp.kernel" or name.startswith("wp.kernel("))

  def _is_func(name: str) -> bool:
    """Check if decorator is wp.func."""
    return name and (name == "wp.func" or name.startswith("wp.func("))

  top_kernels: dict[str, bool] = {}

  # Supports # kernel_analyzer: ignore, ignore[rule], off/on, off[rule]/on[rule]
  ignore_lines: Dict[int, set[str]] = {}
  active_off_rules: set[str] = set()

  for lineno, line in enumerate(source_lines, 1):  # lineno is 1-indexed
    m_off = re.search(r"#\s*kernel_analyzer:\s*off(?:\[(.*?)\])?", line)
    if m_off:
      active_off_rules.add(m_off.group(1).strip() if m_off.group(1) else "*")
    m_on = re.search(r"#\s*kernel_analyzer:\s*on(?:\[(.*?)\])?", line)
    if m_on:
      rule = m_on.group(1).strip() if m_on.group(1) else "*"
      active_off_rules.discard(rule)
      if rule == "*":
        active_off_rules.clear()

    m_ign = re.search(r"#\s*kernel_analyzer:\s*ignore(?:\[(.*?)\])?", line)
    rules_to_ignore = set(active_off_rules)
    if m_ign:
      rules_to_ignore.add(m_ign.group(1).strip() if m_ign.group(1) else "*")

    if rules_to_ignore:
      ignore_lines[lineno] = rules_to_ignore

  def _is_ignored(iss: Issue) -> bool:
    if isinstance(iss.node, ast.FunctionDef):
      start_line = min((d.lineno for d in iss.node.decorator_list), default=iss.node.lineno)
      end_line = iss.node.body[0].lineno - 1 if iss.node.body else iss.node.lineno
    else:
      start_line = iss.node.lineno
      end_line = iss.node.end_lineno or start_line
    matched_rules: set[str] = set()
    for l in range(start_line, end_line + 1):
      if l in ignore_lines:
        matched_rules.update(ignore_lines[l])
    if not matched_rules:
      return False
    if "*" in matched_rules:
      return True
    if isinstance(iss, (MissingDeterministicFactory, MissingDeterministicMaxRecords, AtomicTargetConflict)):
      return any("atomic" in r.lower() or "deterministic" in r.lower() for r in matched_rules)
    return False

  def _is_atomic_call(sub: ast.AST) -> bool:
    return (
      isinstance(sub, ast.Call)
      and isinstance(sub.func, ast.Attribute)
      and isinstance(sub.func.value, ast.Name)
      and sub.func.value.id == "wp"
      and sub.func.attr.startswith("atomic_")
      and sub.func.attr not in ("atomic_or", "atomic_and", "atomic_xor")
    )

  def _is_wp_static(sub: ast.AST) -> bool:
    return (
      isinstance(sub, ast.Call)
      and isinstance(sub.func, ast.Attribute)
      and isinstance(sub.func.value, ast.Name)
      and sub.func.value.id == "wp"
      and sub.func.attr == "static"
    )

  def _extract_root_name(expr: ast.AST) -> Optional[str]:
    root = expr
    while isinstance(root, ast.Subscript):
      root = root.value
    return root.id if isinstance(root, ast.Name) else None

  def _has_loop_break_or_continue(stmts: List[ast.stmt]) -> bool:
    stack: List[ast.AST] = list(stmts)
    while stack:
      cur = stack.pop()
      if isinstance(cur, (ast.Break, ast.Continue)):
        return True
      if isinstance(cur, (ast.For, ast.While, ast.FunctionDef)):
        continue
      stack.extend(ast.iter_child_nodes(cur))
    return False

  def _is_unrolled_for(loop_node: ast.For) -> bool:
    it = loop_node.iter
    if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" and 1 <= len(it.args) <= 3):
      return False
    vals = []
    for a in it.args:
      if isinstance(a, ast.Constant) and isinstance(a.value, int):
        vals.append(a.value)
      elif _is_wp_static(a):
        vals.append(None)
      else:
        return False
    if _has_loop_break_or_continue(loop_node.body):
      return False
    if any(_is_wp_static(sub) for sub in ast.walk(loop_node)):
      return True
    if all(v is not None for v in vals):
      start = 0 if len(vals) == 1 else vals[0]
      end = vals[0] if len(vals) == 1 else vals[1]
      step = 1 if len(vals) < 3 else vals[2]
      if step == 0:
        return False
      return len(range(start, end, step)) <= 16
    return False

  def _find_dynamic_loop_atomics(
    stmts: List[ast.stmt],
    fn_name: str,
    dyn_atomic_funcs: set[str],
    all_atomic_funcs: set[str],
    in_dyn_loop: bool = False,
  ) -> List[ast.Call]:
    found: List[ast.Call] = []
    for stmt in stmts:
      if isinstance(stmt, ast.FunctionDef):
        continue
      cur_dyn = in_dyn_loop
      if isinstance(stmt, ast.While):
        cur_dyn = True
      elif isinstance(stmt, ast.For) and not _is_unrolled_for(stmt):
        cur_dyn = True

      # Check expressions directly on stmt (excluding child statement blocks)
      child_blocks = []
      for field in ("body", "orelse", "finalbody", "handlers"):
        val = getattr(stmt, field, None)
        if isinstance(val, list):
          child_blocks.extend(val)
      child_block_ids = {id(c) for c in child_blocks}

      stack = [stmt]
      while stack:
        cur = stack.pop()
        if isinstance(cur, ast.Call):
          if _is_atomic_call(cur) and not _is_ignored(MissingDeterministicFactory(cur, fn_name)):
            if cur_dyn:
              found.append(cur)
          elif isinstance(cur.func, ast.Name) and not _is_ignored(MissingDeterministicFactory(cur, fn_name)):
            if cur.func.id in dyn_atomic_funcs or (cur_dyn and cur.func.id in all_atomic_funcs):
              found.append(cur)
        for ch in ast.iter_child_nodes(cur):
          if id(ch) not in child_block_ids and not isinstance(ch, ast.FunctionDef):
            stack.append(ch)

      for field in ("body", "orelse", "finalbody"):
        val = getattr(stmt, field, None)
        if isinstance(val, list):
          found.extend(_find_dynamic_loop_atomics(val, fn_name, dyn_atomic_funcs, all_atomic_funcs, cur_dyn))
    return found

  def _get_deterministic_max_records_expr(enclosing_func: ast.FunctionDef) -> Optional[ast.AST]:
    for sub in ast.walk(enclosing_func):
      if isinstance(sub, ast.Assign):
        for target in sub.targets:
          if (
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "deterministic_max_records"
          ):
            return sub.value
      elif isinstance(sub, ast.Dict):
        for k, v in zip(sub.keys, sub.values):
          if isinstance(k, ast.Constant) and k.value == "deterministic_max_records":
            return v
    return None

  def _propagate_func_atomic_info(
    func_map: Dict[str, ast.FunctionDef],
    atomic_funcs: Dict[str, List[ast.Call]],
    dyn_atomic_funcs: set[str],
    atomic_param_indices: Dict[str, set[int]],
    rw_param_indices: Dict[str, set[int]],
  ):
    for fname, fnode in func_map.items():
      param_idx = {a.arg: i for i, a in enumerate(fnode.args.args)}
      atomics = [
        sub for sub in ast.walk(fnode) if _is_atomic_call(sub) and not _is_ignored(MissingDeterministicFactory(sub, fname))
      ]
      if atomics:
        atomic_funcs[fname] = atomics
        for acall in atomics:
          if acall.args:
            rname = _extract_root_name(acall.args[0])
            if rname in param_idx:
              atomic_param_indices.setdefault(fname, set()).add(param_idx[rname])

    changed = True
    while changed:
      changed = False
      for fname, fnode in func_map.items():
        param_idx = {a.arg: i for i, a in enumerate(fnode.args.args)}
        for sub in ast.walk(fnode):
          if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in atomic_funcs:
            if not _is_ignored(MissingDeterministicFactory(sub, fname)):
              if fname not in atomic_funcs:
                atomic_funcs[fname] = [sub]
                changed = True
              for pidx in atomic_param_indices.get(sub.func.id, ()):
                if pidx < len(sub.args):
                  rname = _extract_root_name(sub.args[pidx])
                  if rname in param_idx and param_idx[rname] not in atomic_param_indices.get(fname, ()):
                    atomic_param_indices.setdefault(fname, set()).add(param_idx[rname])
                    changed = True

    for fname, fnode in func_map.items():
      param_idx = {a.arg: i for i, a in enumerate(fnode.args.args)}
      atomic_arg_ids: set[int] = set()
      for sub in ast.walk(fnode):
        if isinstance(sub, ast.Call) and not _is_ignored(MissingDeterministicFactory(sub, fname)):
          if _is_atomic_call(sub) and sub.args:
            for n in ast.walk(sub.args[0]):
              atomic_arg_ids.add(id(n))
          elif isinstance(sub.func, ast.Name) and sub.func.id in atomic_param_indices:
            for pidx in atomic_param_indices[sub.func.id]:
              if pidx < len(sub.args):
                for n in ast.walk(sub.args[pidx]):
                  atomic_arg_ids.add(id(n))
      for sub in ast.walk(fnode):
        if isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Subscript):
          rname = _extract_root_name(sub.target)
          if rname in param_idx:
            rw_param_indices.setdefault(fname, set()).add(param_idx[rname])
        elif isinstance(sub, ast.Subscript) and id(sub) not in atomic_arg_ids:
          rname = _extract_root_name(sub)
          if rname in param_idx:
            rw_param_indices.setdefault(fname, set()).add(param_idx[rname])

    changed = True
    while changed:
      changed = False
      for fname, fnode in func_map.items():
        param_idx = {a.arg: i for i, a in enumerate(fnode.args.args)}
        for sub in ast.walk(fnode):
          if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in rw_param_indices:
            for pidx in rw_param_indices[sub.func.id]:
              if pidx < len(sub.args):
                rname = _extract_root_name(sub.args[pidx])
                if rname in param_idx and param_idx[rname] not in rw_param_indices.get(fname, ()):
                  rw_param_indices.setdefault(fname, set()).add(param_idx[rname])
                  changed = True

    changed = True
    while changed:
      changed = False
      for fname, fnode in func_map.items():
        if fname in dyn_atomic_funcs or fname not in atomic_funcs:
          continue
        if _find_dynamic_loop_atomics(fnode.body, fname, dyn_atomic_funcs, set(atomic_funcs.keys())):
          dyn_atomic_funcs.add(fname)
          changed = True

  # Pre-scan top-level @wp.func definitions for un-ignored atomic operations so
  # calling @wp.kernel functions are checked for a deterministic factory.
  top_level_funcs: Dict[str, ast.FunctionDef] = {}
  top_level_atomic_funcs: Dict[str, List[ast.Call]] = {}
  top_level_dyn_atomic_funcs: set[str] = set()
  top_level_atomic_param_indices: Dict[str, set[int]] = {}
  top_level_rw_param_indices: Dict[str, set[int]] = {}
  called_top_level_atomic_funcs: set[str] = set()
  det_factory_read_in_indices: Dict[str, set[int]] = {}
  det_factory_atomic_out_indices: Dict[str, set[int]] = {}
  if check_atomic:
    for node in ast.iter_child_nodes(tree):
      if isinstance(node, ast.FunctionDef):
        for d in node.decorator_list:
          dec = ast.get_source_segment(source, d)
          if _is_func(dec):
            top_level_funcs[node.name] = node
            break

    _propagate_func_atomic_info(
      top_level_funcs,
      top_level_atomic_funcs,
      top_level_dyn_atomic_funcs,
      top_level_atomic_param_indices,
      top_level_rw_param_indices,
    )

  def _build_static_branch_map(root_node: ast.FunctionDef) -> Dict[int, Dict[int, bool]]:
    branch_map: Dict[int, Dict[int, bool]] = {}

    def _visit(n: ast.AST, current: Dict[int, bool]):
      branch_map[id(n)] = current
      if isinstance(n, ast.If) and any(_is_wp_static(s) for s in ast.walk(n.test)):
        if_id = id(n)
        _visit(n.test, current)
        body_ctx = dict(current)
        body_ctx[if_id] = True
        for b in n.body:
          _visit(b, body_ctx)
        else_ctx = dict(current)
        else_ctx[if_id] = False
        for o in n.orelse:
          _visit(o, else_ctx)
        return
      for ch in ast.iter_child_nodes(n):
        if not isinstance(ch, ast.FunctionDef):
          _visit(ch, current)

    _visit(root_node, {})
    return branch_map

  def _analyze_function(node: ast.FunctionDef, is_nested: bool, enclosing_func: Optional[ast.FunctionDef] = None):
    """Analyze a function definition for kernel issues."""
    # Recursively check nested functions first
    for child in ast.iter_child_nodes(node):
      if isinstance(child, ast.FunctionDef):
        _analyze_function(child, is_nested=True, enclosing_func=node)

    # Find wp.kernel or wp.func decorator
    decorator = None
    for d in node.decorator_list:
      decorator = ast.get_source_segment(source, d)
      if _is_kernel(decorator) or _is_func(decorator):
        break
    else:
      return  # not a kernel/func

    # Nested wp.kernel must have module="unique" (wp.func doesn't need it)
    if _is_kernel(decorator):
      has_unique = 'module="unique"' in decorator or "module='unique'" in decorator
      if is_nested:
        if not has_unique:
          issues.append(MissingModuleUnique(node, node.name))
      else:
        top_kernels[node.name] = has_unique

    # Kernels and functions with atomics must be inside a deterministic factory.
    if check_atomic and (_is_kernel(decorator) or _is_func(decorator)):
      has_det = False
      if is_nested and enclosing_func is not None:
        has_det = any(a.arg == "deterministic" for a in enclosing_func.args.args + enclosing_func.args.kwonlyargs)
      if _is_kernel(decorator) or is_nested:
        for sub in ast.walk(node):
          if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in top_level_atomic_funcs:
            called_top_level_atomic_funcs.add(sub.func.id)
            if not has_det:
              issues.append(MissingDeterministicFactory(sub, node.name))
          elif not has_det and _is_atomic_call(sub):
            issues.append(MissingDeterministicFactory(sub, node.name))

      if is_nested and _is_kernel(decorator) and has_det and enclosing_func is not None:
        # Collect all nested @wp.func / @wp.func_native helpers inside enclosing_func
        local_func_map: Dict[str, ast.FunctionDef] = {}
        local_atomic_funcs: Dict[str, List[ast.Call]] = dict(top_level_atomic_funcs)
        local_dyn_atomic_funcs: set[str] = set(top_level_dyn_atomic_funcs)
        local_atomic_param_indices: Dict[str, set[int]] = {k: set(v) for k, v in top_level_atomic_param_indices.items()}
        local_rw_param_indices: Dict[str, set[int]] = {k: set(v) for k, v in top_level_rw_param_indices.items()}
        sync_func_names: set[str] = set()
        for sub_fn in ast.walk(enclosing_func):
          if isinstance(sub_fn, ast.FunctionDef) and sub_fn is not enclosing_func and sub_fn is not node:
            for sd in sub_fn.decorator_list:
              sdec = ast.get_source_segment(source, sd) or ""
              if "func_native" in sdec and ("WP_TILE_SYNC" in sdec or "__syncthreads" in sdec):
                sync_func_names.add(sub_fn.name)
              if _is_func(sdec):
                local_func_map[sub_fn.name] = sub_fn

        if local_func_map:
          _propagate_func_atomic_info(
            local_func_map,
            local_atomic_funcs,
            local_dyn_atomic_funcs,
            local_atomic_param_indices,
            local_rw_param_indices,
          )

        local_atomic_func_names = set(local_atomic_funcs.keys())
        max_rec_expr = _get_deterministic_max_records_expr(enclosing_func)
        dyn_atomics = _find_dynamic_loop_atomics(node.body, node.name, local_dyn_atomic_funcs, local_atomic_func_names)
        if max_rec_expr is None:
          issues.append(
            MissingDeterministicMaxRecords(
              node,
              node.name,
              'deterministic factory must set "deterministic_max_records" in module_options',
            )
          )
        elif (
          dyn_atomics
          and isinstance(max_rec_expr, ast.Constant)
          and isinstance(max_rec_expr.value, int)
          and max_rec_expr.value <= 1
        ):
          for dcall in dyn_atomics:
            issues.append(
              MissingDeterministicMaxRecords(
                dcall,
                node.name,
                'executes atomic operations in a dynamic loop; "deterministic_max_records" must be set to the loop upper bound (> 1)',
              )
            )

        # Check for thread sync or intra-kernel read / write / AugAssign on atomic target arrays
        direct_atomics = [
          sub for sub in ast.walk(node) if _is_atomic_call(sub) and not _is_ignored(MissingDeterministicFactory(sub, node.name))
        ]
        func_atomic_calls = [
          sub
          for sub in ast.walk(node)
          if isinstance(sub, ast.Call)
          and isinstance(sub.func, ast.Name)
          and sub.func.id in local_atomic_func_names
          and not _is_ignored(MissingDeterministicFactory(sub, node.name))
        ]
        all_unignored_atomics = list(direct_atomics) + func_atomic_calls
        if sync_func_names and all_unignored_atomics:
          has_sync_call = any(
            isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in sync_func_names
            for sub in ast.walk(node)
          )
          if has_sync_call:
            for acall in all_unignored_atomics:
              issues.append(
                AtomicTargetConflict(
                  acall,
                  node.name,
                  "uses thread synchronization with deterministic scatter atomics, which are deferred until kernel exit",
                )
              )

        atomic_ops_by_target: Dict[str, List[ast.Call]] = {}
        atomic_arg0_ids: set[int] = set()
        for acall in direct_atomics:
          if acall.args:
            arg0 = acall.args[0]
            for n in ast.walk(arg0):
              atomic_arg0_ids.add(id(n))
            rname = _extract_root_name(arg0)
            if rname:
              atomic_ops_by_target.setdefault(rname, []).append(acall)

        for fcall in func_atomic_calls:
          if isinstance(fcall.func, ast.Name):
            for pidx in local_atomic_param_indices.get(fcall.func.id, ()):
              if pidx < len(fcall.args):
                arg_expr = fcall.args[pidx]
                for n in ast.walk(arg_expr):
                  atomic_arg0_ids.add(id(n))
                rname = _extract_root_name(arg_expr)
                if rname:
                  atomic_ops_by_target.setdefault(rname, []).append(fcall)

        if atomic_ops_by_target:
          in_params = [p.arg for p in node.args.args if not (p.arg.endswith("_out") or p.arg in ("res", "out"))]
          out_params = [p.arg for p in node.args.args if p.arg.endswith("_out") or p.arg in ("res", "out")]
          read_param_names: set[str] = set()
          for sub in ast.walk(node):
            if isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Load) and id(sub) not in atomic_arg0_ids:
              rname = _extract_root_name(sub)
              if rname:
                read_param_names.add(rname)
            elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in local_rw_param_indices:
              for pidx in local_rw_param_indices[sub.func.id]:
                if pidx < len(sub.args):
                  rname = _extract_root_name(sub.args[pidx])
                  if rname:
                    read_param_names.add(rname)
          det_factory_read_in_indices[enclosing_func.name] = {i for i, p in enumerate(in_params) if p in read_param_names}
          det_factory_atomic_out_indices[enclosing_func.name] = {
            i for i, p in enumerate(out_params) if p in atomic_ops_by_target
          }

          static_branch_map = _build_static_branch_map(node)

          def _conflicts_with_atomics(sub_node: ast.AST, target_name: str) -> bool:
            b_sub = static_branch_map.get(id(sub_node), {})
            for acall in atomic_ops_by_target.get(target_name, ()):
              b_acall = static_branch_map.get(id(acall), {})
              if not any(k in b_acall and b_sub[k] != b_acall[k] for k in b_sub):
                return True
            return False

          aug_target_sub_ids: set[int] = set()
          outer_sub_children_ids: set[int] = set()
          for sub in ast.walk(node):
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Subscript):
              outer_sub_children_ids.add(id(sub.value))

          for sub in ast.walk(node):
            if isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Subscript):
              rname = _extract_root_name(sub.target)
              if rname in atomic_ops_by_target:
                for n in ast.walk(sub.target):
                  aug_target_sub_ids.add(id(n))
                if _conflicts_with_atomics(sub, rname):
                  issues.append(
                    AtomicTargetConflict(
                      sub,
                      node.name,
                      f'modifies "{rname}" in-place alongside atomic operations; deterministic scatter atomics are deferred until kernel exit',
                    )
                  )

          for sub in ast.walk(node):
            if (
              isinstance(sub, ast.Subscript)
              and id(sub) not in atomic_arg0_ids
              and id(sub) not in aug_target_sub_ids
              and id(sub) not in outer_sub_children_ids
            ):
              rname = _extract_root_name(sub)
              if rname in atomic_ops_by_target and _conflicts_with_atomics(sub, rname):
                if isinstance(sub.ctx, ast.Store):
                  issues.append(
                    AtomicTargetConflict(
                      sub,
                      node.name,
                      f'writes "{rname}" directly alongside atomic operations; deterministic scatter atomics are deferred until kernel exit',
                    )
                  )
                elif isinstance(sub.ctx, ast.Load):
                  issues.append(
                    AtomicTargetConflict(
                      sub,
                      node.name,
                      f'reads "{rname}" alongside atomic operations; deterministic scatter atomics are deferred until kernel exit',
                    )
                  )
            elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in local_rw_param_indices:
              for pidx in local_rw_param_indices[sub.func.id]:
                if pidx < len(sub.args):
                  rname = _extract_root_name(sub.args[pidx])
                  if rname in atomic_ops_by_target and _conflicts_with_atomics(sub, rname):
                    issues.append(
                      AtomicTargetConflict(
                        sub,
                        node.name,
                        f'reads or modifies "{rname}" via "{sub.func.id}" alongside atomic operations; deterministic scatter atomics are deferred until kernel exit',
                      )
                    )

    _analyze_kernel(node)

  def _analyze_kernel(node: ast.FunctionDef):
    """Analyze kernel parameters and body."""
    kernel, args = node.name, node.args

    # defaults not allowed
    if args.defaults:
      for param in args.args[-len(args.defaults) :]:
        issues.append(DefaultForbidden(param, kernel))

    # kw defaults not allowed.
    if args.kwonlyargs:
      for param, default in zip(args.kwonlyargs, args.kw_defaults):
        if default:
          issues.append(DefaultForbidden(param, kernel))

    # varargs not allowed.
    if args.vararg:
      issues.append(VarArgsForbidden(args.vararg, kernel))

    # kwargs not allowed.
    if args.kwarg:
      issues.append(KwArgsForbidden(args.kwarg, kernel))

    param_names = set()
    params_ordering = []
    params_multiline = any(a.lineno != args.args[0].lineno for a in args.args)
    param_groups = set()
    param_types = {}
    param_outs = set()
    param_reftypes = set()
    expected_types = {}

    # TODO(team): we might also want to check return types of wp.func
    for param in args.args:
      param_name = param.arg
      param_names.add(param_name)

      # params must be type annotated
      if param.annotation is None:
        issues.append(MissingType(param, kernel))
        continue

      param_type = ast.get_source_segment(source, param.annotation)
      param_type = param_type.replace("types.", "")  # ignore types module prefix
      param_types[param_name] = param_type

      # enforce bracket syntax: wp.array[X] not wp.array(dtype=X)
      if re.search(r"wp\.array\d*d?\(dtype=", param_type):
        issues.append(ParenthesizedArraySyntax(param, kernel))

      has_suffix = param_name.endswith("_in") or param_name.endswith("_out")
      field_name = param_name[: param_name.rfind("_")] if has_suffix else param_name
      param_out = param_name.endswith("_out") or param_name in ("res", "out")
      if param_out:
        param_outs.add(param_name)
      if "array" in param_type:
        param_reftypes.add(param_name)
      param_source, expected_type, param_order = field_info.get(field_name, (None, None, None))

      # TODO(team): indicate array slice with suffix and check type?

      # if parameters are multi-line, parameters must be grouped by comments of the form
      # "{source} {in | out | ""}:" e.g. "Model:" or "Data in:" or "Out:"
      if params_multiline and (param_out, param_source) not in param_groups:
        param_groups.add((param_out, param_source))
        expected_comment = _get_arg_expected_comment(param_source, param_out)
        if param.lineno < 2 or source_lines[param.lineno - 2].strip() != expected_comment:
          issues.append(MissingComment(param, kernel, expected_comment))

      # paramater type must match field type (or generic types if no corresponding field)
      if expected_type is None:
        # if the parameter does not correspond to Model/Data fields, it has no expected type
        # still, there are a few type conventions we stick to
        if "wp.int32" in param_type:
          expected_type = "int"
        elif "wp.float32" in param_type:
          expected_type = "float"
        elif "wp.bool" in param_type:
          expected_type = "bool"
      elif expected_type.startswith("array("):
        # array(...) is our custom annotation that we can translate to wp.array
        expected_dtype = expected_type[expected_type.rfind(" ") + 1 : -1]
        expected_ndim = expected_type.count(",")
        if expected_ndim == 1:
          expected_type = f"wp.array[{expected_dtype}]"
        else:
          expected_type = f"wp.array{expected_ndim}d[{expected_dtype}]"
      elif m := re.match(r"wp\.array(\d?)d?\(dtype=([\w.]+)\)", expected_type):
        # wp.array(dtype=X) or wp.arrayNd(dtype=X) from types.py field specs
        ndim_str, dtype = m.group(1), m.group(2)
        if ndim_str and ndim_str != "1":
          expected_type = f"wp.array{ndim_str}d[{dtype}]"
        else:
          expected_type = f"wp.array[{dtype}]"

      expected_types[param_name] = expected_type
      if expected_type and param_type != expected_type:
        issues.append(TypeMismatch(param, kernel, expected_type, param_source))

      # Model params must not have in/out suffix, Data params must, other params optional
      if param_source == "Model" and has_suffix:
        issues.append(InvalidSuffix(param, kernel, param_source))
      elif param_source == "Data" and not has_suffix:
        issues.append(InvalidSuffix(param, kernel, param_source))

      source_order = {"Model": 0, "Data": 1, None: 2}[param_source]
      param_order = len(params_ordering) if param_order is None else param_order
      params_ordering.append((param_out, source_order, param_order, param_name))

    # parameters must follow a specified order:
    # 1) in fields, then out fields
    # 2) Model, then Data, then other
    # 3) matching Model field order or Data field order
    expected_ordering = sorted(params_ordering)
    if params_ordering != expected_ordering:
      expected_names = [p[3] for p in expected_ordering]
      node.col_offset = node.col_offset + 4
      node.end_lineno = node.lineno
      node.end_col_offset = node.col_offset + len(node.name)
      arg_range = _get_function_arg_range(node, source)
      # Build the full arg strings with comments and types.
      expected_full, prev = [], None
      for e in expected_ordering:
        if prev != e[:2]:
          src = {0: "Model", 1: "Data", 2: None}[e[1]]
          is_out = e[0]
          expected_full.append(_get_arg_expected_comment(src, is_out))
        type_ = expected_types[e[3]] or param_types[e[3]]
        expected_full.append(e[3] + ": " + type_ + ",")
        prev = e[:2]
      issues.append(InvalidParamOrder(node, kernel, expected_names, expected_full, arg_range))

    # don't allow assignments to in parameters
    for sub_node in ast.walk(node):
      # simple assignments:
      if isinstance(sub_node, ast.Assign):
        for target in sub_node.targets:
          assignee = ast.get_source_segment(source, target)
          if assignee in param_names and assignee not in param_outs and assignee in param_reftypes:
            issues.append(InvalidWrite(target, kernel))
      # augmented assignments (+=, -=, etc)
      elif isinstance(sub_node, ast.AugAssign):
        assignee = ast.get_source_segment(source, sub_node.target)
        if assignee in param_names and assignee not in param_outs and assignee in param_reftypes:
          issues.append(InvalidWrite(sub_node.target, kernel))
      # in-place operations like a[i] = value
      elif isinstance(sub_node, ast.Subscript) and isinstance(sub_node.ctx, ast.Store):
        assignee = ast.get_source_segment(source, sub_node.value)
        if assignee in param_names and assignee not in param_outs and assignee in param_reftypes:
          issues.append(InvalidWrite(sub_node.value, kernel))
      # TODO: atomic_add, atomic_sub

    # check *-dimensioned Model fields are indexed with % shape[0]
    star_params = {p for p in param_names if p in star_fields and p in param_reftypes}
    if star_params:

      def _is_modulo_shape0(index_node: ast.AST, param_name: str) -> bool:
        """Check if index is `expr % param.shape[0]`."""
        if not isinstance(index_node, ast.BinOp):
          return False
        if not isinstance(index_node.op, ast.Mod):
          return False
        # right side must be param.shape[0]
        rhs = index_node.right
        if not isinstance(rhs, ast.Subscript):
          return False
        rhs_src = ast.get_source_segment(source, rhs)
        return rhs_src == f"{param_name}.shape[0]"

      # Pass 1: find precomputed safe indices
      # e.g., `gear_id = worldid % actuator_gear.shape[0]`
      safe_indices: Dict[str, str] = {}  # var_name -> param_name
      for sub_node in ast.walk(node):
        if isinstance(sub_node, ast.Assign) and len(sub_node.targets) == 1:
          target = sub_node.targets[0]
          if isinstance(target, ast.Name) and isinstance(sub_node.value, ast.BinOp):
            for sp in star_params:
              if _is_modulo_shape0(sub_node.value, sp):
                safe_indices[target.id] = sp
                break

      # Pass 2: check all subscript accesses on *-dimensioned params
      for sub_node in ast.walk(node):
        if not isinstance(sub_node, ast.Subscript):
          continue
        # check if the subscript target is a *-dimensioned param
        if not isinstance(sub_node.value, ast.Name):
          continue
        param_name = sub_node.value.id
        if param_name not in star_params:
          continue

        # get the first index (the batch/world dimension)
        idx = sub_node.slice
        if isinstance(idx, ast.Tuple) and idx.elts:
          first_idx = idx.elts[0]
        else:
          first_idx = idx

        # check: inline modulo pattern
        if _is_modulo_shape0(first_idx, param_name):
          continue

        # check: precomputed safe index variable
        if isinstance(first_idx, ast.Name) and safe_indices.get(first_idx.id) == param_name:
          continue

        issues.append(MissingBatchModulo(sub_node, kernel, param_name))

  # Analyze all top-level function definitions
  for node in ast.iter_child_nodes(tree):
    if isinstance(node, ast.FunctionDef):
      _analyze_function(node, is_nested=False)

  if check_atomic:
    for fname, atomics in top_level_atomic_funcs.items():
      if fname not in called_top_level_atomic_funcs:
        for sub in atomics:
          issues.append(MissingDeterministicFactory(sub, fname))

    for sub_node in ast.walk(tree):
      if (
        isinstance(sub_node, ast.Call)
        and isinstance(sub_node.func, ast.Attribute)
        and isinstance(sub_node.func.value, ast.Name)
        and sub_node.func.value.id == "wp"
        and sub_node.func.attr in ("launch", "launch_tiled")
        and sub_node.args
        and isinstance(sub_node.args[0], ast.Call)
        and isinstance(sub_node.args[0].func, ast.Name)
      ):
        fname = sub_node.args[0].func.id
        if fname in det_factory_atomic_out_indices and fname in det_factory_read_in_indices:
          inputs_arg = None
          outputs_arg = None
          for kw in sub_node.keywords:
            if kw.arg == "inputs":
              inputs_arg = kw.value
            elif kw.arg == "outputs":
              outputs_arg = kw.value
          if isinstance(inputs_arg, (ast.List, ast.Tuple)) and isinstance(outputs_arg, (ast.List, ast.Tuple)):
            for out_idx in det_factory_atomic_out_indices[fname]:
              if out_idx < len(outputs_arg.elts):
                out_elt = outputs_arg.elts[out_idx]
                out_src = (ast.get_source_segment(source, out_elt) or "").strip()
                for in_idx in det_factory_read_in_indices[fname]:
                  if in_idx < len(inputs_arg.elts):
                    in_src = (ast.get_source_segment(source, inputs_arg.elts[in_idx]) or "").strip()
                    if out_src and out_src == in_src:
                      issues.append(
                        AtomicTargetConflict(
                          out_elt,
                          fname,
                          f'passes "{out_src}" as both a read input and an atomic output; deterministic scatter atomics are deferred until kernel exit',
                        )
                      )
                      break

  # Check operator precedence disambiguation throughout the file
  for sub_node in ast.walk(tree):
    # 1. Compare with bitwise operator
    if isinstance(sub_node, ast.Compare):
      for child in [sub_node.left] + sub_node.comparators:
        if isinstance(child, ast.BinOp) and isinstance(child.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
          if not _is_parenthesized(source_lines, child):
            issues.append(AmbiguousPrecedence(child, "", "parenthesize bitwise operation inside comparison"))

    # 2. Logical not with bitwise operator
    elif isinstance(sub_node, ast.UnaryOp) and isinstance(sub_node.op, ast.Not):
      if isinstance(sub_node.operand, ast.BinOp) and isinstance(sub_node.operand.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        if not _is_parenthesized(source_lines, sub_node.operand):
          issues.append(AmbiguousPrecedence(sub_node, "", 'parenthesize bitwise operation after "not"'))

    # 3. Bitwise shift inside bitwise AND/OR/XOR
    elif isinstance(sub_node, ast.BinOp) and isinstance(sub_node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
      for child in (sub_node.left, sub_node.right):
        if isinstance(child, ast.BinOp) and isinstance(child.op, (ast.LShift, ast.RShift)):
          if not _is_parenthesized(source_lines, child):
            issues.append(AmbiguousPrecedence(child, "", "parenthesize bitwise shift inside bitwise operation"))

    # 4. Bitwise NOT (~) in boolean condition
    elif isinstance(sub_node, (ast.If, ast.While, ast.Assert)):
      tests = [sub_node.test]
      if isinstance(sub_node.test, ast.BoolOp):
        tests = sub_node.test.values
      for t in tests:
        if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Invert):
          issues.append(BitwiseInversionInBoolean(t, ""))

    # wp.* calls inside wp.static(...)
    elif _is_wp_call(sub_node) and sub_node.func.attr == "static":
      for inner in ast.walk(sub_node):
        if inner is not sub_node and _is_wp_call(inner):
          issues.append(WarpCallInStatic(inner, ""))

  # Check wp.launch and wp.launch_tiled calls with custom block_dim
  for sub_node in ast.walk(tree):
    if isinstance(sub_node, ast.Call):
      func = sub_node.func
      if (
        isinstance(func, ast.Attribute)
        and func.attr in ("launch", "launch_tiled")
        and isinstance(func.value, ast.Name)
        and func.value.id == "wp"
      ):
        has_launch_bdim = any(kw.arg == "block_dim" for kw in sub_node.keywords)
        if has_launch_bdim:
          first_arg = (
            sub_node.args[0] if sub_node.args else next((kw.value for kw in sub_node.keywords if kw.arg == "kernel"), None)
          )
          if isinstance(first_arg, ast.Name):
            target_name = first_arg.id
            if target_name in top_kernels and not top_kernels[target_name]:
              issues.append(SharedModuleCustomBlockDim(sub_node, target_name))

  filtered_issues = [i for i in issues if not _is_ignored(i)]
  filter_count, ignore_count = len(filtered_issues), len(issues) - len(filtered_issues)

  logging.info(f"Finished analyzing {filename}. Found {filter_count} issues, ignoring {ignore_count} issues.")

  return filtered_issues
