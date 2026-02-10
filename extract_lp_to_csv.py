#!/usr/bin/env python3
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SECTION_ORDER = [
    "Problem name",
    "Maximize",
    "Minimize",
    "Subject To",
    "Bounds",
    "Generals",
    "Binaries",
    "End",
]

SECTION_PATTERN = re.compile(r"^(Problem name|Maximize|Minimize|Subject To|Bounds|Generals|Binaries|End)\b")


def _collect_notebook_text(nb_path: Path) -> str:
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    chunks: List[str] = []
    for cell in data.get("cells", []):
        source = cell.get("source", [])
        if isinstance(source, list):
            chunks.extend(source)
            chunks.append("\n")
        elif isinstance(source, str):
            chunks.append(source)
            chunks.append("\n")
        for output in cell.get("outputs", []) or []:
            if isinstance(output, dict):
                text = output.get("text")
                if isinstance(text, list):
                    chunks.extend(text)
                    chunks.append("\n")
                elif isinstance(text, str):
                    chunks.append(text)
                    chunks.append("\n")
                data_obj = output.get("data")
                if isinstance(data_obj, dict):
                    for _, value in data_obj.items():
                        if isinstance(value, list):
                            chunks.extend(value)
                            chunks.append("\n")
                        elif isinstance(value, str):
                            chunks.append(value)
                            chunks.append("\n")
    return "".join(chunks)


def _find_lp_block(text: str) -> List[str]:
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("Problem name"):
            start_idx = idx
            break
    if start_idx is None:
        return []
    block: List[str] = []
    for line in lines[start_idx:]:
        block.append(line)
        if line.strip() == "End":
            break
    return block


def _split_sections(block_lines: List[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current = None
    for line in block_lines:
        match = SECTION_PATTERN.match(line.strip())
        if match:
            current = match.group(1)
            sections.setdefault(current, [])
            if line.strip() != current:
                sections[current].append(line.strip())
            continue
        if current:
            sections[current].append(line)
    return sections


def _find_mdl_block(text: str) -> List[str]:
    """Find code block that constructs a docplex Model (mdl = Model(...))."""
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    start_idx = None
    varname = None
    for idx, line in enumerate(lines):
        m = re.search(r"\b(\w+)\s*=\s*Model\s*\(", line)
        if m:
            varname = m.group(1)
            start_idx = idx
            break
        # fallback: bare Model(...) without assignment
        if re.search(r"\bModel\s*\(", line):
            varname = 'mdl'
            start_idx = idx
            break
    if start_idx is None:
        return []
    block: List[str] = []
    # gather a fairly large window after model creation; model definitions often span multiple cells
    window = lines[start_idx: start_idx + 800]
    for line in window:
        block.append(line)
        # stop early if we reach common QP/solve markers
        if 'from_docplex_mp' in line or re.search(r"\bqp\s*=.*from_docplex_mp", line) or 'admm.solve' in line or re.search(r"\bqp\s*=" , line):
            break
    return block


def _split_mdl_sections(block_lines: List[str]) -> Dict[str, List[str]]:
    """Extract logical sections from a docplex `mdl` code block."""
    text = "\n".join(block_lines)
    sections: Dict[str, List[str]] = {}

    # Problem name
    m = re.search(r"(\w+)?\s*=\s*Model\(\s*['\"]([^'\"]+)['\"]", text)
    model_var = None
    if m:
        model_var = m.group(1)
        sections.setdefault('Problem name', []).append(m.group(2))
    else:
        # try bare Model("name")
        m2 = re.search(r"Model\(\s*['\"]([^'\"]+)['\"]", text)
        if m2:
            sections.setdefault('Problem name', []).append(m2.group(1))
    if m:
        sections.setdefault('Problem name', []).append(m.group(1))

    # Objective
    mo = re.search(r"mdl\.maximize\((.*?)\)", text, re.S)
    if mo:
        sections.setdefault('Maximize', []).append(mo.group(1).strip())
    else:
        mo2 = re.search(r"mdl\.minimize\((.*?)\)", text, re.S)
        if mo2:
            sections.setdefault('Minimize', []).append(mo2.group(1).strip())

    # Constraints: collect all mdl.add_constraint(...) occurrences
    cons = re.findall(r"mdl\.add_constraint\((.*?)\)", text, re.S)
    if cons:
        # Clean and add each constraint
        cleaned = [c.strip().replace('\n', ' ') for c in cons]
        sections.setdefault('Subject To', []).extend(cleaned)

    # Variables: capture lines that define integer_var, binary_var, continuous_var
    if model_var:
        var_pattern = rf"(\w+)\s*=\s*{re.escape(model_var)}\.(integer_var|binary_var|continuous_var)\([^\)]*\)"
    else:
        # try common names including 'mdl' or any variable calling .integer_var
        var_pattern = r"(\w+)\s*=\s*(?:\w+)\.(integer_var|binary_var|continuous_var)\([^\)]*\)"
    vars_found = re.findall(var_pattern, text)
    for varname, vtype in vars_found:
        if vtype == 'integer_var':
            sections.setdefault('Generals', []).append(f"{varname}")
        elif vtype == 'binary_var':
            sections.setdefault('Binaries', []).append(f"{varname}")
        else:
            sections.setdefault('Bounds', []).append(f"{varname} (continuous)")

    # Fallback: if no explicit vars found, collect any lines with 'mdl.integer_var' etc.
    if 'Generals' not in sections and 'Binaries' not in sections:
        for line in block_lines:
            if '.integer_var' in line:
                sections.setdefault('Generals', []).append(line.strip())
            if '.binary_var' in line:
                sections.setdefault('Binaries', []).append(line.strip())

    return sections


def _write_csv(sections: Dict[str, List[str]], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "content"])
        for name in SECTION_ORDER:
            if name in sections:
                content = "\n".join([ln.rstrip() for ln in sections[name]]).strip()
                writer.writerow([name, content])
        for name, lines in sections.items():
            if name not in SECTION_ORDER:
                content = "\n".join([ln.rstrip() for ln in lines]).strip()
                writer.writerow([name, content])


def main() -> int:
    # Usage:
    # 1) Process a single notebook: python extract_lp_to_csv.py path/to/notebook.ipynb
    # 2) Process a directory (recursive): python extract_lp_to_csv.py path/to/QISKIT
    if len(sys.argv) < 2:
        # default to local QISKIT folder if available
        base = Path("QISKIT")
        if not base.exists():
            print("Usage: extract_lp_to_csv.py <notebook.ipynb|directory>\nOr run from repository root so that ./QISKIT exists.")
            return 2
        targets = list(base.rglob("*.ipynb"))
    else:
        p = Path(sys.argv[1])
        if not p.exists():
            print(f"Path not found: {p}")
            return 2
        if p.is_dir():
            targets = list(p.rglob("*.ipynb"))
        else:
            targets = [p]

    if not targets:
        print("No .ipynb files found to process.")
        return 1

    def process_notebook(nb_path: Path) -> Tuple[bool, str]:
        try:
            text = _collect_notebook_text(nb_path)
            block = _find_lp_block(text)
            sections = {}
            if block:
                sections = _split_sections(block)
            else:
                # try fallback: parse docplex Model(...) construction
                mdl_block = _find_mdl_block(text)
                if not mdl_block:
                    return False, "LP block not found"
                sections = _split_mdl_sections(mdl_block)
            out_path = nb_path.with_suffix('.csv')
            _write_csv(sections, out_path)
            return True, str(out_path)
        except Exception as e:
            return False, f"error: {e}"

    processed = 0
    failures: List[Tuple[str, str]] = []
    for nb in sorted(targets):
        ok, msg = process_notebook(nb)
        if ok:
            print(f"Wrote: {msg}")
            processed += 1
        else:
            print(f"Skipped {nb}: {msg}")
            failures.append((str(nb), msg))

    print(f"Processed {processed}/{len(targets)} notebooks.")
    if failures:
        print("Failures:")
        for fn, reason in failures:
            print(f" - {fn}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
