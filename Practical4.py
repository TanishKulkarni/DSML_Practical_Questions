# build_decision_tree_id3.py
# ID3 decision tree implementation from scratch (no external libraries).
# Reads a CSV (categorical or numeric), prints step-by-step entropy and info gain,
# builds the tree and prints the ROOT node attribute.
#
# Usage: place cosmetics.csv in same folder and run:
#    python build_decision_tree_id3.py

import csv
import math
from collections import defaultdict, Counter
from copy import deepcopy

CSV_FILE = "venv\Datasets\Lipstick.csv"   # change if your file has a different name
TARGET_COL = None            # if None, script uses last column as target

# ---------------------------
# Utilities
# ---------------------------
def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = [h.strip() for h in rows[0]]
    data = [dict(zip(header, [cell.strip() for cell in row])) for row in rows[1:]]
    return header, data

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False

def maybe_convert_numeric(data, header):
    """
    Convert columns that are numeric across all rows to floats.
    Returns set of numeric columns.
    """
    numeric_cols = set()
    for col in header:
        # check every non-empty value is numeric
        vals = [row[col] for row in data if row[col] != ""]
        if len(vals) > 0 and all(is_number(v) for v in vals):
            numeric_cols.add(col)
            for row in data:
                if row[col] == "":
                    row[col] = None
                else:
                    row[col] = float(row[col])
    return numeric_cols

def discretize_numeric_column(data, col):
    """Simple discretization: split by median into '<=med' and '>med'"""
    vals = [row[col] for row in data if row[col] is not None]
    if not vals:
        return
    med = sorted(vals)[len(vals)//2]
    for row in data:
        if row[col] is None:
            row[col] = "missing"
        else:
            row[col] = "<=%.6g" % med if row[col] <= med else ">%.6g" % med
    return med

# ---------------------------
# Entropy & Info Gain
# ---------------------------
def entropy(rows, target):
    counts = Counter(r[target] for r in rows)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p) if p>0 else 0
    return ent

def info_gain(rows, attr, target):
    base_entropy = entropy(rows, target)
    # get value partitions
    subsets = defaultdict(list)
    for r in rows:
        subsets[r[attr]].append(r)
    total = len(rows)
    remainder = 0.0
    for vrows in subsets.values():
        remainder += (len(vrows)/total) * entropy(vrows, target)
    gain = base_entropy - remainder
    return gain

# ---------------------------
# ID3 Tree builder
# ---------------------------
def majority_class(rows, target):
    counts = Counter(r[target] for r in rows)
    return counts.most_common(1)[0][0]

def id3(rows, attributes, target, depth=0, max_depth=None):
    # If all rows have same target, return leaf
    targets = set(r[target] for r in rows)
    if len(targets) == 1:
        return {"type":"leaf", "class": next(iter(targets))}
    if not attributes or (max_depth is not None and depth >= max_depth):
        return {"type":"leaf", "class": majority_class(rows, target)}

    # compute info gain for each attribute
    gains = {}
    for attr in attributes:
        gains[attr] = info_gain(rows, attr, target)

    # choose best attribute
    best_attr = max(gains, key=gains.get)
    if gains[best_attr] <= 1e-12:
        # no informative attribute -> leaf
        return {"type":"leaf", "class": majority_class(rows, target)}

    tree = {"type":"node", "attribute": best_attr, "children": {}}

    # split dataset and recurse
    values = set(r[best_attr] for r in rows)
    for val in values:
        subset = [r for r in rows if r[best_attr] == val]
        if not subset:
            tree["children"][val] = {"type":"leaf", "class": majority_class(rows, target)}
        else:
            # remaining attributes exclude best_attr
            new_attrs = [a for a in attributes if a != best_attr]
            tree["children"][val] = id3(subset, new_attrs, target, depth+1, max_depth)
    return tree

def print_tree(node, indent=""):
    if node["type"] == "leaf":
        print(indent + "-> Leaf: class =", node["class"])
    else:
        print(indent + f"[Attribute: {node['attribute']}]")
        for val, child in node["children"].items():
            print(indent + f"  If {node['attribute']} == {val}:")
            print_tree(child, indent + "    ")

# ---------------------------
# Main
# ---------------------------
def main():
    header, data = read_csv(CSV_FILE)
    global TARGET_COL
    if TARGET_COL is None:
        TARGET_COL = header[-1]
    print("Header columns:", header)
    print("Using target column:", TARGET_COL)

    # Convert numeric-like columns to numeric types
    numeric_cols = maybe_convert_numeric(data, header)
    if numeric_cols:
        print("Detected numeric columns:", numeric_cols)
        # discretize numeric columns into two bins so ID3 can handle them
        for col in list(numeric_cols):
            med = discretize_numeric_column(data, col)
            print(f"Discretized numeric column '{col}' by median = {med}")

    # list of attributes (exclude target)
    attributes = [c for c in header if c != TARGET_COL]
    print("\nAttributes considered:", attributes)

    # Print dataset preview
    print("\nSample rows (first 8):")
    for r in data[:8]:
        print(r)

    # Compute and print base entropy of target
    base_ent = entropy(data, TARGET_COL)
    print(f"\nBase entropy H({TARGET_COL}) = {base_ent:.6f}")

    # Compute info gain for each attribute and show step by step
    gains = {}
    print("\nInfo gains (step-by-step):")
    for attr in attributes:
        g = info_gain(data, attr, TARGET_COL)
        gains[attr] = g

        # print value partitions and entropies
        subsets = defaultdict(list)
        for r in data:
            subsets[r[attr]].append(r)
        print(f"\nAttribute: {attr}")
        for v, rows in subsets.items():
            h = entropy(rows, TARGET_COL)
            print(f"  Value '{v}': count={len(rows)}, entropy={h:.6f}")
        print(f"  => Information Gain for {attr} = {g:.6f}")

    # Determine root
    root_attr = max(gains, key=gains.get)
    print(f"\n=> ROOT NODE (attribute with highest information gain): {root_attr}")
    print(f"   Information Gain = {gains[root_attr]:.6f}")

    # Optionally build full tree and print (comment out if not required)
    print("\nBuilding full decision tree (ID3)...\n")
    tree = id3(data, attributes, TARGET_COL)
    print_tree(tree)

if __name__ == "__main__":
    main()
