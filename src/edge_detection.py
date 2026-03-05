import numpy as np
import cv2

HORIZONTAL = 1
VERTICAL = -1

def gaussian_smoothing(image, ksize=5, sigma=1.0):
    smoothed = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return smoothed

def compute_gradient(image):
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    G = np.abs(dx) + np.abs(dy)
    D = -np.sign(np.abs(dx) - np.abs(dy))
    return G, D


def compute_anchors(G, D, anchor_threshold=8, scan_interval=1):
    h, w = G.shape
    anchors = []
    for y in range(1, h-1, scan_interval):
        for x in range(1, w-1, scan_interval):
            if G[y, x] == 0:
                continue
            if D[y, x] == HORIZONTAL:
                if (
                    G[y, x] - G[y-1, x] >= anchor_threshold and
                    G[y, x] - G[y+1, x] >= anchor_threshold
                ):
                    anchors.append((y, x))

            elif D[y, x] == VERTICAL:
                if (
                    G[y, x] - G[y, x-1] >= anchor_threshold and
                    G[y, x] - G[y, x+1] >= anchor_threshold
                ):
                    anchors.append((y, x))

    return anchors


def edge_drawing(G, D, anchors, grad_threshold=36):
    h, w = G.shape
    visited = np.zeros_like(G, dtype=bool)
    edges = []
    def walk(y, x, step):
        segment = []
        while True:

            if not (1 <= y < h-1 and 1 <= x < w-1):
                break
            if visited[y, x]:
                break

            if G[y, x] < grad_threshold:
                break
            visited[y, x] = True
            segment.append((y, x))

            if D[y, x] == HORIZONTAL:

                candidates = [
                    (y-1, x+step),
                    (y,   x+step),
                    (y+1, x+step)
                ]
            else:
                candidates = [
                    (y+step, x-1),
                    (y+step, x),
                    (y+step, x+1)
                ]

            best = None
            best_grad = 0
            for ny, nx in candidates:
                if visited[ny, nx]:
                    continue
                g = G[ny, nx]
                if g > best_grad:
                    best_grad = g
                    best = (ny, nx)

            if best is None:
                break

            y, x = best
        return segment


    for ay, ax in anchors:
        if visited[ay, ax]:
            continue

        seg1 = walk(ay, ax, +1)
        seg2 = walk(ay, ax, -1)
        segment = list(reversed(seg2)) + seg1
        if len(segment) > 5:
            edges.append(segment)

    return edges