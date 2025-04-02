from typing import List
import math
from geometry.basic import Point, orientation


def convex_hull(points: List[Point]) -> List[Point]:
    """
    Compute the convex hull of a set of points using Graham's scan algorithm.

    Args:
        points: List of points

    Returns:
        List of points forming the convex hull
    """
    n = len(points)

    # Need at least 3 points for a convex hull
    if n < 3:
        return points

    # Find the bottom-most point (and left-most in case of tie)
    bottom_idx = 0
    for i in range(1, n):
        if points[i].y < points[bottom_idx].y or (
                points[i].y == points[bottom_idx].y and points[i].x < points[bottom_idx].x):
            bottom_idx = i

    # Swap the bottom-most point with the first point
    points[0], points[bottom_idx] = points[bottom_idx], points[0]

    # Sort points by polar angle with respect to the bottom-most point
    p0 = points[0]

    def compare(p1, p2):
        o = orientation(p0, p1, p2)

        if o == 0:
            # Collinear points, sort by distance from p0
            dist1 = (p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2
            dist2 = (p2.x - p0.x) ** 2 + (p2.y - p0.y) ** 2
            return -1 if dist1 < dist2 else 1

        return -1 if o == 2 else 1

    import functools
    points[1:] = sorted(points[1:], key=functools.cmp_to_key(compare))

    # Remove collinear points with the first point (keep the farthest)
    m = 1
    for i in range(1, n):
        while i < n - 1 and orientation(p0, points[i], points[i + 1]) == 0:
            i += 1
        points[m] = points[i]
        m += 1

    # If we have less than 3 points, convex hull is not possible
    if m < 3:
        return points[:m]

    # Build the convex hull
    hull = [points[0], points[1], points[2]]

    for i in range(3, m):
        # Remove points that make a non-left turn
        while len(hull) > 1 and orientation(hull[-2], hull[-1], points[i]) != 2:
            hull.pop()

        hull.append(points[i])

    return hull


def jarvis_march(points: List[Point]) -> List[Point]:
    """
    Compute the convex hull of a set of points using the Jarvis march algorithm (gift wrapping).

    Args:
        points: List of points

    Returns:
        List of points forming the convex hull
    """
    n = len(points)

    # Need at least 3 points for a convex hull
    if n < 3:
        return points

    # Find the leftmost point
    leftmost = 0
    for i in range(1, n):
        if points[i].x < points[leftmost].x:
            leftmost = i

    hull = []
    p = leftmost
    q = 0

    # Loop to find the convex hull
    while True:
        hull.append(points[p])

        q = (p + 1) % n

        for i in range(n):
            # If point i is more counterclockwise than q, update q
            if orientation(points[p], points[i], points[q]) == 2:
                q = i

        p = q

        # Complete the hull
        if p == leftmost:
            break

    return hull


def graham_scan(points: List[Point]) -> List[Point]:
    """
    Compute the convex hull of a set of points using the Graham scan algorithm.

    Args:
        points: List of points

    Returns:
        List of points forming the convex hull
    """
    # Same as convex_hull function, but keeping the name for clarity
    return convex_hull(points)


def quick_hull(points: List[Point]) -> List[Point]:
    """
    Compute the convex hull of a set of points using the Quick Hull algorithm.

    Args:
        points: List of points

    Returns:
        List of points forming the convex hull
    """
    if len(points) <= 3:
        return convex_hull(points)  # Use simpler algorithm for small inputs

    # Find the leftmost and rightmost points
    min_x = min(points, key=lambda p: p.x)
    max_x = max(points, key=lambda p: p.x)

    # Initialize hull with leftmost and rightmost points
    hull = {min_x, max_x}

    # Split points into two sets
    left_of_line = []
    right_of_line = []

    for point in points:
        if point in hull:
            continue

        # Determine which side of the line the point is on
        o = orientation(min_x, max_x, point)

        if o == 2:  # Point is to the right of the line
            right_of_line.append(point)
        elif o == 1:  # Point is to the left of the line
            left_of_line.append(point)

    # Recursively find hull points
    _find_hull(min_x, max_x, right_of_line, hull)
    _find_hull(max_x, min_x, left_of_line, hull)

    # Convert to list and order the hull points
    hull_list = list(hull)

    # Find the bottom-most point
    bottom_idx = 0
    for i in range(1, len(hull_list)):
        if hull_list[i].y < hull_list[bottom_idx].y or (
                hull_list[i].y == hull_list[bottom_idx].y and hull_list[i].x < hull_list[bottom_idx].x):
            bottom_idx = i

    # Sort by polar angle from the bottom-most point
    p0 = hull_list[bottom_idx]

    def compare(p1, p2):
        o = orientation(p0, p1, p2)
        if o == 0:
            dist1 = (p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2
            dist2 = (p2.x - p0.x) ** 2 + (p2.y - p0.y) ** 2
            return -1 if dist1 < dist2 else 1
        return -1 if o == 2 else 1

    import functools
    hull_list.sort(key=functools.cmp_to_key(compare))

    return hull_list


def _find_hull(p1: Point, p2: Point, points: List[Point], hull: set):
    """
    Helper function for Quick Hull algorithm.

    Args:
        p1, p2: Two points that form a line
        points: List of points to consider
        hull: Set of hull points being constructed
    """
    if not points:
        return

    # Find point with maximum distance from line p1-p2
    max_dist = 0
    farthest_point = None

    for point in points:
        # Calculate distance from point to line p1-p2
        dist = abs((p2.y - p1.y) * point.x - (p2.x - p1.x) * point.y + p2.x * p1.y - p2.y * p1.x) / \
               math.sqrt((p2.y - p1.y) ** 2 + (p2.x - p1.x) ** 2)

        if dist > max_dist:
            max_dist = dist
            farthest_point = point

    if farthest_point is None:
        return

    # Add farthest point to hull
    hull.add(farthest_point)

    # Split points into two sets
    points1 = []  # Points outside triangle p1-farthest-p2
    points2 = []  # Points outside triangle p2-farthest-p1

    for point in points:
        if point == farthest_point:
            continue

        o1 = orientation(p1, farthest_point, point)
        o2 = orientation(farthest_point, p2, point)

        if o1 == 2:  # Point is to the right of line p1-farthest
            points1.append(point)
        elif o2 == 2:  # Point is to the right of line farthest-p2
            points2.append(point)

    # Recursively process the two sets
    _find_hull(p1, farthest_point, points1, hull)
    _find_hull(farthest_point, p2, points2, hull)


def monotone_chain(points: List[Point]) -> List[Point]:
    """
    Compute the convex hull of a set of points using the Monotone Chain algorithm.

    Args:
        points: List of points

    Returns:
        List of points forming the convex hull
    """
    # Sort the points lexicographically
    points.sort(key=lambda p: (p.x, p.y))

    if len(points) <= 2:
        return points

    # Build lower hull
    lower = []
    for point in points:
        while len(lower) >= 2 and orientation(lower[-2], lower[-1], point) != 2:
            lower.pop()
        lower.append(point)

    # Build upper hull
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and orientation(upper[-2], upper[-1], point) != 2:
            upper.pop()
        upper.append(point)

    # Remove duplicate points where lower and upper hulls meet
    return lower[:-1] + upper[:-1]