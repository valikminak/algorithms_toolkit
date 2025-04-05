from typing import List, Tuple, Optional
import math
from kosmos.faw.geometry.basic import Point, orientation, on_segment


def line_segment_intersection(p1: Point, q1: Point, p2: Point, q2: Point) -> Optional[Point]:
    """
    Find the intersection point of two line segments.

    Args:
        p1, q1: First line segment from p1 to q1
        p2, q2: Second line segment from p2 to q2

    Returns:
        Intersection point if the line segments intersect, None otherwise
    """
    # Find the orientations
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case: segments intersect if orientations are different
    if o1 != o2 and o3 != o4:
        # Calculate the intersection point
        a1 = q1.y - p1.y
        b1 = p1.x - q1.x
        c1 = a1 * p1.x + b1 * p1.y

        a2 = q2.y - p2.y
        b2 = p2.x - q2.x
        c2 = a2 * p2.x + b2 * p2.y

        determinant = a1 * b2 - a2 * b1

        if abs(determinant) < 1e-9:  # Lines are parallel
            return None

        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant

        return Point(x, y)

    # Special cases: collinear points
    if o1 == 0 and on_segment(p1, p2, q1):
        return p2
    if o2 == 0 and on_segment(p1, q2, q1):
        return q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return p1
    if o4 == 0 and on_segment(p2, q1, q2):
        return q1

    return None  # No intersection


def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
    """
    Check if a point is inside a polygon using the ray casting algorithm.

    Args:
        point: The point to check
        polygon: List of points forming the polygon

    Returns:
        True if the point is inside the polygon, False otherwise
    """
    n = len(polygon)

    if n < 3:
        return False

    # Check if the point is on an edge
    for i in range(n):
        j = (i + 1) % n
        if on_segment(polygon[i], point, polygon[j]):
            return True

    # Ray casting algorithm
    inside = False
    for i in range(n):
        j = (i + 1) % n

        # Check if ray from point to the right intersects with polygon edge
        if ((polygon[i].y > point.y) != (polygon[j].y > point.y)) and \
                (point.x < (polygon[j].x - polygon[i].x) * (point.y - polygon[i].y) /
                 (polygon[j].y - polygon[i].y) + polygon[i].x):
            inside = not inside

    return inside


def closest_pair_of_points(points: List[Point]) -> Tuple[Point, Point, float]:
    """
    Find the closest pair of points in a set of points.

    Args:
        points: List of points

    Returns:
        Tuple of (point1, point2, min_distance)
    """
    if len(points) < 2:
        raise ValueError("At least two points are required")

    # Sort points by x-coordinate
    points_x = sorted(points, key=lambda p: p.x)

    # Helper function to calculate distance
    def distance(p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    # Divide and conquer algorithm
    def closest_pair_recursive(points_x, points_y):
        n = len(points_x)

        # Base case: if we have 2 or 3 points, compute minimum distance directly
        if n <= 3:
            min_dist = float('inf')
            min_pair = (None, None)

            for i in range(n):
                for j in range(i + 1, n):
                    dist = distance(points_x[i], points_x[j])
                    if dist < min_dist:
                        min_dist = dist
                        min_pair = (points_x[i], points_x[j])

            return min_pair[0], min_pair[1], min_dist

        # Divide points into two halves
        mid = n // 2
        mid_point = points_x[mid]

        # Divide points_x into left and right halves
        points_x_left = points_x[:mid]
        points_x_right = points_x[mid:]

        # Divide points_y into left and right halves
        points_y_left = []
        points_y_right = []

        for p in points_y:
            if p.x <= mid_point.x:
                points_y_left.append(p)
            else:
                points_y_right.append(p)

        # Recursively find closest pair in left and right halves
        p1_left, p2_left, dist_left = closest_pair_recursive(points_x_left, points_y_left)
        p1_right, p2_right, dist_right = closest_pair_recursive(points_x_right, points_y_right)

        # Find the smaller distance
        if dist_left < dist_right:
            p1, p2, min_dist = p1_left, p2_left, dist_left
        else:
            p1, p2, min_dist = p1_right, p2_right, dist_right

        # Create a strip of points whose x-distance from mid_point is less than min_dist
        strip = []
        for p in points_y:
            if abs(p.x - mid_point.x) < min_dist:
                strip.append(p)

        # Find the closest pair in the strip
        for i in range(len(strip)):
            # Check at most 7 points ahead
            j = i + 1
            while j < len(strip) and (strip[j].y - strip[i].y) < min_dist:# geometry/intersection.py (continuing the closest_pair_of_points function)
                dist = distance(strip[i], strip[j])
                if dist < min_dist:
                    min_dist = dist
                    p1, p2 = strip[i], strip[j]
                j += 1

        return p1, p2, min_dist

    # Sort points by y-coordinate
    points_y = sorted(points, key=lambda p: p.y)

    # Call the recursive function
    return closest_pair_recursive(points_x, points_y)