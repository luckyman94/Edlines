import numpy as np
import cv2

def fit_line(points):
    pts = np.array(points)
    xs = pts[:,1]
    ys = pts[:,0]

    A = np.vstack([xs, np.ones(len(xs))]).T
    b, a = np.linalg.lstsq(A, ys, rcond=None)[0]

    return a, b

def point_line_distance(p, a, b):
    y, x = p
    return abs(y - (a + b*x)) / np.sqrt(1 + b*b)

def compute_line_angle(line):
    y1, x1 = line[0]
    y2, x2 = line[-1]
    dy = y2 - y1
    dx = x2 - x1
    return np.arctan2(dy, dx)

def line_fit_chain(chain, min_len=10, threshold=1.0):
    lines = []
    i = 0
    n = len(chain)
    while i + min_len < n:
        segment = chain[i:i+min_len]
        a, b = fit_line(segment)
        err = max(point_line_distance(p, a, b) for p in segment)
        if err > threshold:
            i += 1
            continue

        j = i + min_len
        while j < n:
            d = point_line_distance(chain[j], a, b)
            if d > threshold:
                break
            j += 1
        a, b = fit_line(chain[i:j]) 
        lines.append(chain[i:j])
        i = j
    return lines


def detect_lines(edges):
    lines = []
    for edge in edges:
        segments = line_fit_chain(edge)
        lines.extend(segments)
    return lines