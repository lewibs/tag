import math
def euclidian(x1,y1,x2,y2):
    return math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2))

def get_angle_in_radians(start, end):
    """
    Calculate the angle in radians from start to end.

    Parameters:
    start (tuple): The starting point (x1, y1).
    end (tuple): The ending point (x2, y2).

    Returns:
    float: The angle in radians.
    """
    x1, y1 = start
    x2, y2 = end

    # Calculate the differences
    dx = x2 - x1
    dy = y2 - y1

    # Compute the angle in radians
    angle = math.atan2(dy, dx)
    
    return angle

def points_on_circle(cx, cy, radius, num_points=100):
    """
    Generate unique points on the boundary of a circle, rounded to the nearest integer.

    Parameters:
    cx (float): x-coordinate of the center of the circle.
    cy (float): y-coordinate of the center of the circle.
    radius (float): Radius of the circle.
    num_points (int): Number of points to generate on the circle boundary.

    Returns:
    list of tuples: List of (x, y) points on the boundary of the circle.
    """
    points = [] # Use a set to avoid duplicates
    angle_step = 2 * math.pi / num_points

    for i in range(num_points):
        angle = i * angle_step
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)

        # Round to the nearest integer and convert to tuple
        point = (int(x), int(y))
        
        points.append(point)  # Add to set to ensure uniqueness

    # Convert set to list
    return list(points)