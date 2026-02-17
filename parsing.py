import os
import re
import sys
from collections import defaultdict

def parse_log_file(filepath):
    category_stats = []
    primal_bound = None

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Match FPR category lines
            match = re.match(
                r"^\s*(\S+)\s+(\S+)\s+(-?\d+)\s+(-?\d+)",
                line
            )
            if match:
                ranker = match.group(1)
                chooser = match.group(2)
                num_inc = int(match.group(3))
                num_feas = int(match.group(4))
                category_stats.append((ranker, chooser, num_inc, num_feas))

            # Match primalBound line
            match_primal = re.search(
                r"primalBound\s*=\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)",
                line
            )
            if match_primal:
                primal_bound = float(match_primal.group(1))

    return category_stats, primal_bound


def main(directory):
    aggregated = defaultdict(lambda: {"NumInc": 0, "NumFeasible": 0})
    primal_bounds = []

    print("\nPER-FILE PRIMAL BOUNDS")
    print("=" * 60)

    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)

        if not os.path.isfile(filepath):
            continue

        category_stats, primal = parse_log_file(filepath)

        # Aggregate category stats
        for ranker, chooser, num_inc, num_feas in category_stats:
            key = (ranker, chooser)
            aggregated[key]["NumInc"] += num_inc
            aggregated[key]["NumFeasible"] += num_feas

        # Print per-file primal bound
        if primal is not None:
            primal_bounds.append(primal)
            print(f"{filename:40} primalBound = {primal}")
        else:
            print(f"{filename:40} primalBound = NOT FOUND")

        # Check for TYPE RANDOM special case
        type_random_inc = None
        other_inc_positive = False
        for ranker, chooser, num_inc, _ in category_stats:
            if ranker == "TYPE" and chooser == "RANDOM":
                type_random_inc = num_inc
            elif num_inc > 0:
                other_inc_positive = True

        mark = ""
        if type_random_inc is not None and type_random_inc == 0 and other_inc_positive:
            mark = " <-- TYPE RANDOM failed but others succeeded"
            print(mark)

    print("\nAGGREGATED FPR STATS (Across All Files)")
    print("=" * 60)
    for (ranker, chooser), stats in sorted(aggregated.items()):
        print(f"{ranker:12} {chooser:12} "
              f"NumInc={stats['NumInc']:8} "
              f"NumFeasible={stats['NumFeasible']:8}")

    print("\nPRIMAL BOUND SUMMARY")
    print("=" * 60)
    if primal_bounds:
        print(f"Files with primalBound : {len(primal_bounds)}")
    else:
        print("No primalBound values found.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_logs.py <directory>")
        sys.exit(1)

    main(sys.argv[1])
