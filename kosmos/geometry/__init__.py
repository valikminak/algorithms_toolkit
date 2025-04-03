# geometry/__init__.py

from kosmos.geometry.basic import Point, orientation
from kosmos.geometry.convex_hull import convex_hull, jarvis_march, graham_scan, quick_hull
from kosmos.geometry.intersection import (
    line_segment_intersection, point_in_polygon, closest_pair_of_points
)

__all__ = [
    # Basic
    'Point', 'orientation',

    # Convex Hull
    'convex_hull', 'jarvis_march', 'graham_scan', 'quick_hull',

    # Intersection
    'line_segment_intersection', 'point_in_polygon', 'closest_pair_of_points'
]