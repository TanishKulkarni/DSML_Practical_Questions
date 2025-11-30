import math

# Given points
points = {
    "p1": (0.1, 0.6),
    "p2": (0.15, 0.71),
    "p3": (0.08, 0.9),
    "p4": (0.16, 0.85),
    "p5": (0.2, 0.3),
    "p6": (0.25, 0.5),
    "p7": (0.24, 0.1),
    "p8": (0.3, 0.2),
}

# Initial centroids 
m1 = points["p1"]
m2 = points["p8"]

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# Assign each point to nearest centroid 
assignments = {}
for name, coord in points.items():
    d1 = euclidean(coord, m1)
    d2 = euclidean(coord, m2)
    cluster = 1 if d1 <= d2 else 2
    assignments[name] = cluster

# print assignments
print("Print assignments (nearest centroid):")
for name in sorted(points.keys(), key=lambda x: int(x[1:])):
    print(f" {name}: cluster {assignments[name]}")
# 1) Which cluster does p6 belong to?
print("\nAnswer 1) P6 belongs to cluster", assignments["p6"])

# 2) Population of the cluster around m2
cluster2_points= [name for name, c in assignments.items() if c == 2]
print("Answer 2) Points in cluster around m2 (Cluster 2):", cluster2_points)
print("          Population (size) of cluster around m2 = ", len(cluster2_points))

# 3) Updated centroids 
def mean_of_points(names_list):
    xs = [points[n][0] for n in names_list]
    ys = [points[n][1] for n in names_list]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

cluster1_points = [name for name, c in assignments.items() if c == 1]

m1_updated = mean_of_points(cluster1_points)
m2_updated = mean_of_points(cluster2_points)

print("\n Answer 3) updated centroids after assignment:")
print(f" Updated m1 (centroid of cluster 1) = ({m1_updated[0]:.6f}, {m2_updated[1]:.6f})")
print(f" Updated m2 (centroid of cluster 2) = ({m2_updated[0]:.6f}, {m2_updated[1]:.6f})")