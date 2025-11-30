import math

points = {
    "P1": (2, 10),
    "P2": (2, 5),
    "P3": (8, 4),
    "P4": (5, 8),
    "P5": (7, 5),
    "P6": (6, 4),
    "P7": (1, 2),
    "P8": (4, 9),
}

# Initial centroids
m1 = points["P1"]
m2 = points["P4"]
m3 = points["P7"]

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

assignments = {}
distances = {}

for name, coord in points.items():
    d1 = euclidean(coord, m1)
    d2 = euclidean(coord, m2)
    d3 = euclidean(coord, m3)
    distances[name] = (d1, d2, d3)
    nearest = min((d1, 1), (d2, 2), (d3, 3))[1]
    assignments[name] = nearest

# print assignments 
for name in sorted(points.keys(), key=lambda x: int(x[1:])):
    d = distances[name]
    print(f" {name}: Cluster {assignments[name]} (distances -> m1:{d[0]:.6f}, m2:{d[1]:.6f}, m3:{d[2]:.6f})")

# 1) Which cluster does P6 belong to?
print("\n1) Which cluster does P6 belong to?")
print(f"   P6 belongs to Cluster {assignments['P6']}")

# 2) Population of the cluster around m3 (Cluster 3)
cluster3_points = [name for name, c in assignments.items() if c == 3]
print("\n2) Population of cluster around m3 (Cluster 3):")
print(f"   Points in Cluster 3: {cluster3_points}")
print(f"   Population (size) = {len(cluster3_points)}")

# 3) Updated centroids (mean of points in each cluster)
def mean_of_points(names_list):
    xs = [points[n][0] for n in names_list]
    ys = [points[n][1] for n in names_list]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

cluster1_points = [name for name, c in assignments.items() if c == 1]
cluster2_points = [name for name, c in assignments.items() if c == 2]

m1_updated = mean_of_points(cluster1_points)
m2_updated = mean_of_points(cluster2_points)
m3_updated = mean_of_points(cluster3_points)

print("\n3) Updated centroids after one assignment step:")
print(f"   Cluster 1 points: {cluster1_points}")
print(f"   Updated m1 = ({m1_updated[0]:.6f}, {m1_updated[1]:.6f})")

print(f"   Cluster 2 points: {cluster2_points}")
print(f"   Updated m2 = ({m2_updated[0]:.6f}, {m2_updated[1]:.6f})")

print(f"   Cluster 3 points: {cluster3_points}")
print(f"   Updated m3 = ({m3_updated[0]:.6f}, {m3_updated[1]:.6f})")