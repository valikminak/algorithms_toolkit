from typing import List, Tuple, Union, Optional
import math


class Point:
    """Point in 2D space."""

    def __init__(self, x: float, y: float):
        """
        Initialize a point with x and y coordinates.

        Args:
            x: X-coordinate
            y: Y-coordinate
        """
        self.x = float(x)
        self.y = float(y)

    def distance_to(self, other: 'Point') -> float:
        """
        Calculate Euclidean distance to another point.

        Args:
            other: Another point

        Returns:
            The Euclidean distance between the points
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def manhattan_distance(self, other: 'Point') -> float:
        """
        Calculate Manhattan distance to another point.

        Args:
            other: Another point

        Returns:
            The Manhattan distance between the points
        """
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __eq__(self, other: object) -> bool:
        """Check if two points are equal."""
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __repr__(self) -> str:
        """String representation of the point."""
        return f"Point({self.x}, {self.y})"

    def __hash__(self) -> int:
        """Hash value for the point."""
        return hash((self.x, self.y))


class Vector:
    """2D vector with basic vector operations."""

    def __init__(self, x: float, y: float):
        """
        Initialize a vector with x and y components.

        Args:
            x: X-component
            y: Y-component
        """
        self.x = float(x)
        self.y = float(y)

    @classmethod
    def from_points(cls, p1: Point, p2: Point) -> 'Vector':
        """
        Create a vector from two points (p1 to p2).

        Args:
            p1: Starting point
            p2: Ending point

        Returns:
            A vector from p1 to p2
        """
        return cls(p2.x - p1.x, p2.y - p1.y)

    def magnitude(self) -> float:
        """
        Calculate the magnitude (length) of the vector.

        Returns:
            The magnitude of the vector
        """
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self) -> 'Vector':
        """
        Return a normalized version of the vector (unit vector).

        Returns:
            A unit vector in the same direction
        """
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0)
        return Vector(self.x / mag, self.y / mag)

    def dot(self, other: 'Vector') -> float:
        """
        Calculate the dot product with another vector.

        Args:
            other: Another vector

        Returns:
            The dot product of the two vectors
        """
        return self.x * other.x + self.y * other.y

    def cross(self, other: 'Vector') -> float:
        """
        Calculate the cross product with another vector.

        Args:
            other: Another vector

        Returns:
            The cross product of the two vectors
        """
        return self.x * other.y - self.y * other.x

    def __add__(self, other: 'Vector') -> 'Vector':
        """Add two vectors."""
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector') -> 'Vector':
        """Subtract two vectors."""
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector':
        """Multiply vector by a scalar."""
        return Vector(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> 'Vector':
        """Divide vector by a scalar."""
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        return Vector(self.x / scalar, self.y / scalar)

    def __repr__(self) -> str:
        """String representation of the vector."""
        return f"Vector({self.x}, {self.y})"


def orientation(p: Point, q: Point, r: Point) -> int:
    """
    Determine the orientation of three points.

    Args:
        p, q, r: Three points

    Returns:
        0 if collinear, 1 if clockwise, 2 if counterclockwise
    """
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)

    if abs(val) < 1e-9:  # Account for floating point errors
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise


def distance(p1: Point, p2: Point) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        p1, p2: Two points

    Returns:
        The Euclidean distance between the points
    """
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def angle_between_points(p1: Point, p2: Point, p3: Point) -> float:
    """
    Calculate the angle between three points (angle at p2).

    Args:
        p1, p2, p3: Three points

    Returns:
        The angle in radians
    """
    a = distance(p2, p3)
    b = distance(p1, p3)
    c = distance(p1, p2)

    # Law of cosines
    cos_angle = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)

    # Clamp to avoid domain errors due to floating point imprecision
    cos_angle = min(1, max(-1, cos_angle))

    return math.acos(cos_angle)


def polar_angle(p: Point, ref: Point = None) -> float:
    """
    Calculate the polar angle of a point relative to a reference point.

    Args:
        p: The point
        ref: Reference point (defaults to origin)

    Returns:
        The polar angle in radians
    """
    if ref is None:
        ref = Point(0, 0)

    return math.atan2(p.y - ref.y, p.x - ref.x)


def on_segment(p: Point, q: Point, r: Point) -> bool:
    """
    Check if point q lies on line segment pr.

    Args:
        p, q, r: Three points

    Returns:
        True if q lies on segment pr, False otherwise
    """
    return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
            q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y) and
            orientation(p, q, r) == 0)


def area_of_triangle(p1: Point, p2: Point, p3: Point) -> float:
    """
    Calculate the area of a triangle formed by three points.

    Args:
        p1, p2, p3: Three points

    Returns:
        The area of the triangle
    """
    return abs((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2.0)


def centroid(points: List[Point]) -> Point:
    """
    Calculate the centroid of a set of points.

    Args:
        points: List of points

    Returns:
        The centroid point
    """
    if not points:
        raise ValueError("Cannot compute centroid of empty list of points")

    x_sum = sum(p.x for p in points)
    y_sum = sum(p.y for p in points)

    return Point(x_sum / len(points), y_sum / len(points))