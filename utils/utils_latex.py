from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


def _format_cell(x) -> str:
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        if abs(x - round(x)) <= 1e-12:
            return str(int(round(x)))
        return f"{x:.4e}"
    return str(x)


def save_table_latex(
    filename: str | Path,
    headers: Sequence[str],
    rows: Iterable[Sequence],
    caption: str,
    label: str,
) -> None:
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\begin{tabular}{" + "c" * len(headers) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(_format_cell(x) for x in row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")


def save_table_csv(
    filename: str | Path,
    headers: Sequence[str],
    rows: Iterable[Sequence],
) -> None:
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(headers))
        writer.writerows(rows)


def plot_gap_distribution(gaps: Sequence[float], filename: str | Path, bins: int = 30) -> None:
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(gaps, bins=bins)
    plt.xlabel("gap_rand")
    plt.ylabel("frequency")
    plt.title("Gap Distribution (Closed-form vs Random)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
