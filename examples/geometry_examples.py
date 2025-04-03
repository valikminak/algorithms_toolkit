import random

from kosmos.geometry import graham_scan, quick_hull
from kosmos.geometry import line_segment_intersection, point_in_polygon, closest_pair_of_points
from kosmos.geometry import Point, orientation
from kosmos.geometry.basic import distance


def basic_geometry_examples():
    """Example usage of basic geometric functions."""
    p1 = Point(0, 0)
    p2 = Point(4, 0)
    p3 = Point(4, 3)

    print("Points:")
    print(f"p1: {p1}")
    print(f"p2: {p2}")
    print(f"p3: {p3}")

    # Calculate distance
    dist = distance(p1, p3)
    print(f"\nDistance between {p1} and {p3}: {dist}")

    # Check orientation
    o = orientation(p1, p2, p3)
    orientation_str = "collinear" if o == 0 else "clockwise" if o == 1 else "counterclockwise"
    print(f"Orientation of points {p1}, {p2}, {p3}: {orientation_str}")


def convex_hull_examples():
    """Example usage of convex hull algorithms."""
    # Generate random points
    random.seed(42)
    points = [Point(random.randint(0, 100), random.randint(0, 100)) for _ in range(20)]

    print("Convex Hull Algorithms:")

    # Graham's scan
    hull = graham_scan(points)
    print(f"Graham's scan found {len(hull)} hull points")

    # Quick Hull
    hull = quick_hull(points)
    print(f"Quick Hull found {len(hull)} hull points")

    # Visualize
    print("\nVisualizing convex hull (see plot)...")
    # Uncomment to see visualization
    # visualize_convex_hull(points, hull)


def intersection_examples():
    """Example usage of intersection algorithms."""
    # Line segment intersection
    p1 = Point(1, 1)
    q1 = Point(10, 10)
    p2 = Point(1, 10)
    q2 = Point(10, 1)

    intersection = line_segment_intersection(p1, q1, p2, q2)

    print("Line Segment Intersection:")
    if intersection:
        print(f"Lines intersect at {intersection}")
    else:
        print("Lines do not intersect")

    # Point in polygon
    polygon = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
    point1 = Point(5, 5)
    point2 = Point(15, 15)

    print("\nPoint in Polygon:")
    print(f"Is {point1} inside the polygon? {point_in_polygon(point1, polygon)}")
    print(f"Is {point2} inside the polygon? {point_in_polygon(point2, polygon)}")

    # Closest pair of points
    random.seed(42)
    points = [Point(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]

    p1, p2, min_dist = closest_pair_of_points(points)

    print("\nClosest Pair of Points:")
    print(f"Points {p1} and {p2} with distance {min_dist}")


def run_all_examples():
    """Run all geometry examples."""
    print("=" * 50)
    print("BASIC GEOMETRY EXAMPLES")
    print("=" * 50)
    basic_geometry_examples()

    print("\n" + "=" * 50)
    print("CONVEX HULL EXAMPLES")
    print("=" * 50)
    convex_hull_examples()

    print("\n" + "=" * 50)
    print("INTERSECTION EXAMPLES")
    print("=" * 50)
    intersection_examples()


if __name__ == "__main__":
    run_all_examples()