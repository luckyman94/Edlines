import math
from scipy.special import comb
from scipy.stats import binom

def compute_nfa(n, k, img_shape, p=0.125):
    N = max(img_shape[0], img_shape[1])
    N4 = N ** 4
    tail = binom.sf(k - 1, n, p)
    return N4 * tail


def pixel_angle(gx, gy):
    return math.atan2(gy, gx)


def validate_line(segment, gx_map, gy_map, img_shape,
                  angle_tolerance=math.pi / 8, eps=1.0):
    n = len(segment)
    if n == 0:
        return False

    y0, x0 = segment[0]
    y1, x1 = segment[-1]
    seg_angle = math.atan2(y1 - y0, x1 - x0)

    k = 0
    for (y, x) in segment:
        px_angle = math.atan2(gy_map[y, x], gx_map[y, x])
        
        edge_angle = px_angle + math.pi / 2
        
        diff = (seg_angle - edge_angle) % math.pi
        diff = min(diff, math.pi - diff)
        
        if diff <= angle_tolerance:
            k += 1

    nfa = compute_nfa(n, k, img_shape)
    return nfa <= eps


def validate_lines(line_segments, gx_map, gy_map, img_shape, eps=1.0):
    valid = []
    for seg in line_segments:
        if validate_line(seg, gx_map, gy_map, img_shape, eps=eps):
            valid.append(seg)
    return valid