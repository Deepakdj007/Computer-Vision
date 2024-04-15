import numpy as np

def get_limits(color):
    colors = {
        'blue': ([100, 100, 100], [140, 255, 255]),  # Lower and upper limits for blue
        'red': ([0, 100, 100], [10, 255, 255]),      # Lower and upper limits for red
        'orange': ([5, 100, 100], [20, 255, 255]),   # Lower and upper limits for orange
        'yellow': ([20, 100, 100], [30, 255, 255]),  # Lower and upper limits for yellow
        'green': ([40, 100, 100], [80, 255, 255])    # Lower and upper limits for green
    }

    if color.lower() in colors:
        lower_limit, upper_limit = colors[color.lower()]
        lower_limit = np.array(lower_limit, dtype=np.uint8)
        upper_limit = np.array(upper_limit, dtype=np.uint8)
        return lower_limit, upper_limit
    else:
        print(f"Color '{color}' not supported.")
        return None, None