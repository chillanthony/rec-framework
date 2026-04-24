"""
make_tiny_dataset.py — Sample a tiny subset of a benchmark dataset for debug use.

Reads the three pre-split inter files (.train / .valid / .test) of a benchmark
dataset, randomly picks N users from the test split (where each user appears
exactly once), then filters all three splits to keep only those users.

The result is written to  dataset/<src>-tiny/  using the same filename
convention, so RecBole can load it via benchmark_filename.

Usage
-----
# Default: 200 users, source = amazon-videogames-2023-5c-llo
python scripts/make_tiny_dataset.py

# Custom source and size
python scripts/make_tiny_dataset.py --src amazon-scientific-2018 --n_users 100

# Explicit destination name
python scripts/make_tiny_dataset.py --src amazon-videogames-2023-5c-llo --dst my-tiny

# Reproducible sampling
python scripts/make_tiny_dataset.py --seed 42
"""

import argparse
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a tiny sampled copy of a benchmark dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--src",
        default="amazon-videogames-2023-5c-llo",
        help="Source dataset name (folder under dataset/). Default: amazon-videogames-2023-5c-llo",
    )
    p.add_argument(
        "--dst",
        default=None,
        metavar="NAME",
        help="Destination dataset name. Default: <src>-tiny",
    )
    p.add_argument(
        "--n_users",
        type=int,
        default=200,
        help="Number of users to sample. Default: 200",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed for reproducible sampling. Default: 2024",
    )
    return p.parse_args()


def read_header_and_col(path: Path) -> tuple[str, int]:
    """Return (header_line, index_of_user_id_column)."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
    cols = header.split("\t")
    for i, col in enumerate(cols):
        # Match "user_id:token" or plain "user_id"
        if col.split(":")[0] == "user_id":
            return header, i
    raise ValueError(f"No user_id column found in {path}. Columns: {cols}")


def collect_users(path: Path, user_col: int) -> list[str]:
    """Return list of all unique user ids in file (preserving first-seen order)."""
    seen: dict[str, None] = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            uid = line.split("\t")[user_col]
            seen[uid] = None
    return list(seen.keys())


def filter_split(
    src_path: Path,
    dst_path: Path,
    user_col: int,
    keep_users: set[str],
) -> int:
    """Copy lines whose user_id is in keep_users. Returns number of lines written."""
    written = 0
    with open(src_path, encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        header = fin.readline()
        fout.write(header)
        for line in fin:
            uid = line.split("\t")[user_col]
            if uid in keep_users:
                fout.write(line)
                written += 1
    return written


def main() -> None:
    args = parse_args()
    dst_name = args.dst or f"{args.src}-tiny"

    src_dir = DATASET_DIR / args.src
    dst_dir = DATASET_DIR / dst_name

    if not src_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {src_dir}")

    # Detect inter files — support both benchmark split and single-file formats
    test_file = src_dir / f"{args.src}.test.inter"
    if not test_file.exists():
        # Fall back to a single .inter file
        test_file = src_dir / f"{args.src}.inter"
        if not test_file.exists():
            raise FileNotFoundError(
                f"No .test.inter or .inter file found in {src_dir}"
            )
        is_benchmark = False
    else:
        is_benchmark = True

    # --- Determine user_id column index from test/inter file ---
    header_line, user_col = read_header_and_col(test_file)

    # --- Sample users from test split ---
    all_users = collect_users(test_file, user_col)
    if len(all_users) < args.n_users:
        print(
            f"Warning: requested {args.n_users} users but only {len(all_users)} "
            f"available. Using all."
        )
        sampled = all_users
    else:
        random.seed(args.seed)
        sampled = random.sample(all_users, args.n_users)

    keep: set[str] = set(sampled)
    print(f"Sampled {len(keep)} users (seed={args.seed})")

    # --- Create destination directory ---
    dst_dir.mkdir(parents=True, exist_ok=True)

    # --- Filter and write splits ---
    if is_benchmark:
        for split in ("train", "valid", "test"):
            src_path = src_dir / f"{args.src}.{split}.inter"
            dst_path = dst_dir / f"{dst_name}.{split}.inter"
            if not src_path.exists():
                print(f"  [skip] {src_path.name} not found")
                continue
            n = filter_split(src_path, dst_path, user_col, keep)
            print(f"  {split:5s}: {n:6d} interactions → {dst_path.name}")
    else:
        # Single .inter file
        src_path = src_dir / f"{args.src}.inter"
        dst_path = dst_dir / f"{dst_name}.inter"
        n = filter_split(src_path, dst_path, user_col, keep)
        print(f"  inter: {n:6d} interactions → {dst_path.name}")

    # --- Copy optional mapping files (user2id / item2id) if present ---
    for suffix in ("user2id", "item2id"):
        src_map = src_dir / f"{args.src}.{suffix}"
        if src_map.exists():
            dst_map = dst_dir / f"{dst_name}.{suffix}"
            dst_map.write_bytes(src_map.read_bytes())
            print(f"  {suffix}: copied → {dst_map.name}")

    print(f"\nDone. Tiny dataset written to: {dst_dir}")
    print(f"Use dataset name '{dst_name}' in your config.")


if __name__ == "__main__":
    main()
