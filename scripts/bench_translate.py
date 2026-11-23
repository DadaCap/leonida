#!/usr/bin/env python3
"""Throughput of the translator (tools/rdna2spirv.py): microseconds per instruction, so a slowdown is visible.

    python -I scripts/bench_translate.py                 # fixtures + synthetic programs, best of 3
    python -I scripts/bench_translate.py --json          # the same numbers as JSON
    python -I scripts/bench_translate.py --runs 5 --max-slowdown 1.5

The repository tests what the translator *emits*; nothing measures how long it takes. A change to the parser or the
instruction table can cost a large share of the build time and still pass every test, and about 26,503 shaders are
translated per build. This script is the cheap check for that: it times `rdna2spirv.translate()` on inputs that
need no game data, no LLVM and no SPIRV-Tools:

- `corpus`: every fixture in tests/semantic/fixtures/disasm, the hand-written listings the semantic tests use.
  Pixel and vertex listings translate as they are; compute listings need the FSSM header the tests pass separately,
  so they are counted as skipped.
- `synthetic`: generated straight-line programs (ALU, s_nop) at fixed instruction counts. The ALU program is the
  instruction work, the s_nop program is mostly per-instruction overhead, and the two together separate a slower
  parser from a slower code generator.

Times are best-of-`--runs` seconds of one `translate()` call per program; medians are not used because a single
unlucky run (another process taking the CPU) is what dominates the tail. The numbers are machine-specific: compare
them with a run of the same commit, not with the table in the pull request that added this script.
"""
import argparse
import json
import os
import statistics
import struct
import sys
import tempfile
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
FIXTURES = os.path.join(ROOT, "tests", "semantic", "fixtures", "disasm")
sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS)
import rdna2spirv  # noqa: E402

# A DXBC container with zero parts declares no inputs, which is enough for a listing that reads no attribute
# (tests/semantic/support.py uses the same stub).
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

ALU_BODY = ("\tv_add_f32 v1, v1, 1.0\n"
            "\tv_mul_f32 v2, v1, 2.0\n"
            "\tv_cndmask_b32 v3, v1, v2, vcc\n"
            "\tv_and_b32 v4, v3, 0xff\n")
NOP_BODY = "\ts_nop 0\n"


def synthetic(name, body, instructions):
    """A whole pixel shader: comment, the repeated body, and the export the stage needs to be a shader."""
    lines = [f"; ps_{name} synthetic benchmark program\n"]
    lines.extend([body] * ((instructions + body.count("\n") - 1) // body.count("\n")))
    lines.extend(["\tv_mov_b32 v1, 1\n",
                  "\texp mrt0 v1, v1, v1, v1 done\n",
                  "\ts_waitcnt expcnt(0)\n",
                  "\ts_endpgm\n"])
    return "".join(lines).encode("utf-8")


def translate_bytes(stem, listing, runs):
    """Best of `runs` wall-clock seconds for one translate() of this listing, and the instruction count."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        with open(os.path.join(tmp, "disasm", stem + ".s"), "wb") as f:
            f.write(listing)
        with open(os.path.join(tmp, stem + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        path = os.path.join(tmp, "disasm", stem + ".s")
        best = None
        for _ in range(runs):
            start = time.perf_counter()
            with warnings.catch_warnings():
                # The translator leaves its .dxbc handle to the garbage collector; that is its own business here.
                warnings.simplefilter("ignore")
                rdna2spirv.translate(path)
            elapsed = time.perf_counter() - start
            best = elapsed if best is None else min(best, elapsed)
    return best


def instruction_count(listing):
    return sum(1 for line in listing.decode("utf-8", "replace").splitlines()
               if line.startswith("\t") and line.strip())


def corpus_rows(runs):
    """Time every fixture that translates on its own, and count the ones that need an FSSM header."""
    rows, skipped = [], []
    for name in sorted(os.listdir(FIXTURES)):
        if not name.endswith(".s"):
            continue
        stem = name[:-2]
        if stem.startswith("cs_"):
            # A compute listing reads its group size and RSRC2 out of the FSSM header, which translate() takes from
            # <dir>/../fssm/<stem>.fssm; the semantic tests pass it per fixture (tests/semantic/support.py).
            skipped.append(f"{stem}: compute listing, needs a FSSM header")
            continue
        with open(os.path.join(FIXTURES, name), "rb") as f:
            listing = f.read()
        try:
            seconds = translate_bytes(stem, listing, runs)
        except Exception as error:      # a listing this run cannot translate: report it, never count it as fast
            skipped.append(f"{stem}: {type(error).__name__}: {error}")
            continue
        instructions = instruction_count(listing)
        rows.append({"name": stem, "instructions": instructions, "seconds": seconds,
                     "us_per_instruction": seconds / instructions * 1e6 if instructions else 0.0})
    return rows, skipped


def synthetic_rows(runs, size):
    rows = []
    for name, body, instructions in (("alu", ALU_BODY, size), ("nop", NOP_BODY, size)):
        listing = synthetic(name, body, instructions)
        seconds = translate_bytes("ps_" + name, listing, runs)
        count = instruction_count(listing)
        rows.append({"name": name, "instructions": count, "seconds": seconds,
                     "us_per_instruction": seconds / count * 1e6})
    return rows


def summarise(rows):
    if not rows:
        return {"programs": 0, "instructions": 0, "seconds": 0.0, "us_per_instruction": 0.0}
    instructions = sum(r["instructions"] for r in rows)
    seconds = sum(r["seconds"] for r in rows)
    return {"programs": len(rows), "instructions": instructions, "seconds": seconds,
            "us_per_instruction": seconds / instructions * 1e6 if instructions else 0.0}


# Recorded on this machine (AMD Ryzen 9 desktop, Windows 11, Python 3.12.4, commit 1cc6e3c) with --runs 3: the numbers
# to compare a later run against on the same machine. A different machine moves the absolute value; the shape does not.
BASELINE = {"corpus_us_per_instruction": 55.4, "alu_us_per_instruction": 32.3, "nop_us_per_instruction": 14.0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3, help="translate() calls per program, best one counted")
    parser.add_argument("--size", type=int, default=2000, help="instructions in each synthetic program")
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of the tables")
    parser.add_argument("--max-slowdown", type=float, default=None,
                        help="exit 1 when a per-instruction time exceeds BASELINE by this factor")
    arguments = parser.parse_args(argv)
    if arguments.runs < 1:
        parser.error("--runs must be at least 1")
    if arguments.size < 1:
        parser.error("--size must be at least 1")

    corpus, skipped = corpus_rows(arguments.runs)
    synthetic_programs = synthetic_rows(arguments.runs, arguments.size)
    corpus_summary = summarise(corpus)
    synthetic_summary = summarise(synthetic_programs)

    measured = {"corpus_us_per_instruction": corpus_summary["us_per_instruction"]}
    measured.update({f"{row['name']}_us_per_instruction": row["us_per_instruction"]
                     for row in synthetic_programs})

    if arguments.json:
        print(json.dumps({
            "runs": arguments.runs,
            "corpus": {"summary": corpus_summary, "programs": corpus, "skipped": skipped},
            "synthetic": {"summary": synthetic_summary, "programs": synthetic_programs, "size": arguments.size},
            "baseline": BASELINE,
            "measured": measured,
            "slowdown": {k: (measured[k] / BASELINE[k] if BASELINE.get(k) else None) for k in measured},
        }, indent=2, sort_keys=True))
        return 0

    print(f"corpus: {corpus_summary['programs']} fixtures, {corpus_summary['instructions']} instructions, "
          f"{corpus_summary['seconds']:.3f} s, {corpus_summary['us_per_instruction']:.1f} us/instruction "
          f"({len(skipped)} skipped)")
    worst = sorted(corpus, key=lambda r: -r["us_per_instruction"])[:5]
    for row in worst:
        print(f"  {row['name']:<28} {row['instructions']:>5} instr  {row['seconds'] * 1e3:>8.2f} ms  "
              f"{row['us_per_instruction']:>6.1f} us/instr")
    print(f"\nsynthetic ({arguments.size} instructions):")
    for row in synthetic_programs:
        print(f"  {row['name']:<28} {row['instructions']:>5} instr  {row['seconds'] * 1e3:>8.2f} ms  "
              f"{row['us_per_instruction']:>6.1f} us/instr  "
              f"({measured[row['name'] + '_us_per_instruction'] / BASELINE[row['name'] + '_us_per_instruction']:.2f}x "
              f"the recorded baseline)")
    if skipped:
        print("\nskipped (need the FSSM header, which these runs do not pass):")
        for line in skipped:
            print("  " + line)

    if arguments.max_slowdown is not None:
        slow = {k: v / BASELINE[k] for k, v in measured.items() if BASELINE.get(k) and v > BASELINE[k] * arguments.max_slowdown}
        if slow:
            for name, factor in sorted(slow.items()):
                print(f"slower than baseline by {factor:.2f}x: {name}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
