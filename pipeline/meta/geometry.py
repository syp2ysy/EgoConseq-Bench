"""Frozen-scene geometric collision: a vertical cylinder moving horizontally at a fixed floor height.

The body is a disc of radius r swept in the floor plane and extruded over the open height band
(y0 + clearance, y0 + top), where y0 is the observation's start floor and stays fixed for the whole
motion. Obstacles are the scene triangles clipped to that band and projected onto the floor plane;
a face that only touches a band boundary is no obstacle (a step up to the clearance is climbable,
geometry from the body top upwards is free). Turns happen in place and never change the occupied
disc. A position is in contact when the disc comes within r - TOLERANCE_M of an obstacle edge or
its centre lies inside a projected obstacle face; a forward motion stops at its first contact.
Contacts are solved analytically (disc against segment and vertex), not with polygonal buffers.
"""

from __future__ import annotations

import math

import numpy as np

TOLERANCE_M = 1e-4       # tangency: a disc reaching within r of a face by less than this passes
CONTACT_WINDOW_M = 1e-3  # faces reached within this travel after the first contact are co-contacts
BOUNDARY_M = 1e-9        # a contact exactly at the end of a motion belongs to that motion
ATOM_FORWARD_M = .25


def heading(yaw):
    return np.array([-math.sin(yaw), -math.cos(yaw)])


class Obstacles:
    """Band-clipped scene faces projected to (x, z) edges, with the 3D points they come from."""

    def __init__(self, triangles, lo, hi, faces=None):
        triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
        faces = np.arange(len(triangles)) if faces is None else np.asarray(faces)
        y = triangles[..., 1]
        keep = (y.max(1) > lo) & (y.min(1) < hi)
        triangles, faces = triangles[keep], faces[keep]
        points, valid = [], []
        for i in range(3):  # vertices inside the band
            points.append(triangles[:, i])
            valid.append((triangles[:, i, 1] >= lo) & (triangles[:, i, 1] <= hi))
        for i, j in ((0, 1), (1, 2), (2, 0)):  # edge crossings of the two band planes
            p, q = triangles[:, i], triangles[:, j]
            dy = q[:, 1] - p[:, 1]
            for level in (lo, hi):
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = (level - p[:, 1]) / dy
                ok = (dy != 0) & (t > 0) & (t < 1)
                points.append(p + np.where(ok, t, 0.)[:, None] * (q - p))
                valid.append(ok)
        points, valid = np.stack(points, 1), np.stack(valid, 1)
        count = valid.sum(1)
        points, valid, faces, count = points[count > 0], valid[count > 0], faces[count > 0], count[count > 0]
        # a planar convex polygon projects to a convex polygon: order its points by angle
        xz = points[..., [0, 2]]
        centre = (xz * valid[..., None]).sum(1) / count[:, None]
        angle = np.where(valid, np.arctan2(xz[..., 1] - centre[:, None, 1], xz[..., 0] - centre[:, None, 0]), np.inf)
        order = np.argsort(angle, axis=1, kind="stable")
        points = np.take_along_axis(points, order[..., None], 1)
        xz = points[..., [0, 2]]
        rows, k = np.nonzero(np.arange(points.shape[1])[None, :] < count[:, None])
        nxt = (k + 1) % count[rows]
        self.a, self.b = xz[rows, k], xz[rows, nxt]
        self.a3, self.b3 = points[rows, k], points[rows, nxt]
        self.face = faces[rows]
        self.box_min = np.minimum(self.a, self.b)
        self.box_max = np.maximum(self.a, self.b)
        # faces with area are solid footprints: a centre strictly inside one is in contact
        polygon_xz = np.where((np.arange(points.shape[1])[None, :] < count[:, None])[..., None], xz, np.nan)
        area = np.zeros(len(count))
        for n in range(3, points.shape[1] + 1):
            m = count == n
            if m.any():
                p = xz[m, :n]
                q = np.roll(p, -1, axis=1)
                area[m] = .5 * (p[..., 0] * q[..., 1] - q[..., 0] * p[..., 1]).sum(1)
        solid = np.abs(area) > 1e-12
        self.poly, self.poly_count, self.poly_sign = xz[solid], count[solid], np.sign(area[solid])
        self.poly_face = faces[solid]
        self.poly_min = np.nanmin(polygon_xz[solid], axis=1) if solid.any() else np.zeros((0, 2))
        self.poly_max = np.nanmax(polygon_xz[solid], axis=1) if solid.any() else np.zeros((0, 2))

    def inside(self, p):
        """Projected solid faces whose interior strictly contains p."""
        near = np.flatnonzero(np.all((self.poly_min < p) & (p < self.poly_max), axis=1))
        hits = []
        for i in near:
            n = self.poly_count[i]
            a = self.poly[i, :n]
            b = np.roll(a, -1, axis=0)
            cross = (b[:, 0] - a[:, 0]) * (p[1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (p[0] - a[:, 0])
            if np.all(cross * self.poly_sign[i] > 0):
                hits.append(int(self.poly_face[i]))
        return hits

    def blocked(self, p, radius):
        """Static occupancy of the disc at p, independent of heading: inside a face or within reach of an edge."""
        p = np.asarray(p, dtype=float)
        if self.inside(p):
            return True
        a, b = self.a, self.b
        edge = b - a
        size2 = (edge * edge).sum(1)
        t = np.clip(np.divide(((p - a) * edge).sum(1), size2, out=np.zeros(len(a)), where=size2 > 0), 0., 1.)
        return bool(len(a)) and float(np.min(np.linalg.norm(a + t[:, None] * edge - p, axis=1))) < radius - TOLERANCE_M

    def first_contact(self, p, d, length, radius):
        """First travel s in [0, length] along unit direction d at which the disc reaches an obstacle edge.

        Returns (s, edge indices reached within CONTACT_WINDOW_M of s) or None when the motion is free.
        """
        p, d = np.asarray(p, dtype=float), np.asarray(d, dtype=float)
        reach = radius - TOLERANCE_M
        end = p + length * d
        lo, hi = np.minimum(p, end) - reach, np.maximum(p, end) + reach
        sel = np.flatnonzero(np.all((self.box_max >= lo) & (self.box_min <= hi), axis=1))
        a, b = self.a[sel], self.b[sel]
        s = np.full(len(sel), np.inf)
        for v in (a, b):  # rounded ends: disc centre entering the circle of radius `reach`
            w = v - p
            along = w @ d
            disc = reach * reach - ((w * w).sum(1) - along * along)
            root = np.sqrt(np.maximum(disc, 0.))
            s = np.minimum(s, np.where((disc >= 0) & (along + root > 0), np.maximum(along - root, 0.), np.inf))
        edge = b - a
        size = np.linalg.norm(edge, axis=1)
        flat = size > 0
        u = np.divide(edge, size[:, None], out=np.zeros_like(edge), where=flat[:, None])
        n = np.stack([-u[:, 1], u[:, 0]], 1)
        h0, dh, t0 = (p - a) * n, d @ n.T, (p - a) * u
        h0, t0 = h0.sum(1), t0.sum(1)
        s = np.where(flat & (np.abs(h0) < reach) & (t0 >= 0) & (t0 <= size), 0., s)
        toward = flat & (np.abs(h0) >= reach) & (h0 * dh < 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            enter = (np.where(h0 > 0, reach, -reach) - h0) / dh
        at = ((p + np.where(toward, enter, 0.)[:, None] * d - a) * u).sum(1)
        s = np.minimum(s, np.where(toward & (at >= 0) & (at <= size), enter, np.inf))
        first = float(s.min()) if len(s) else math.inf
        if first > length + BOUNDARY_M:
            return None
        return first, sel[s <= first + CONTACT_WINDOW_M]

    def contact_points(self, edges, centre):
        """3D point of each contacted edge closest (in plan) to the disc centre at contact."""
        a, b = self.a[edges], self.b[edges]
        edge = b - a
        size2 = (edge * edge).sum(1)
        t = np.clip(np.divide(((centre - a) * edge).sum(1), size2, out=np.zeros(len(a)), where=size2 > 0), 0., 1.)
        return self.a3[edges] + t[:, None] * (self.b3[edges] - self.a3[edges])


def poses(start_xz, yaw, atomic):
    """(x, z, yaw) after every atom of an atomic program; positive turn degrees turn right."""
    p, values = np.asarray(start_xz, dtype=float), []
    for atom in atomic:
        if atom["type"] == "forward":
            p = p + atom["m"] * heading(yaw)
        else:
            yaw = yaw - math.radians(atom["deg"])
        values.append((float(p[0]), float(p[1]), float(yaw)))
    return values


def execute(obstacles, start_xz, yaw, atomic, radius):
    """Sweep one atomic program; the first contact ends it. Turns never collide."""
    p = np.asarray(start_xz, dtype=float)
    if obstacles.blocked(p, radius):
        return {"valid_start": False}
    for index, atom in enumerate(atomic):
        if atom["type"] != "forward":
            yaw = yaw - math.radians(atom["deg"])
            continue
        d = heading(yaw)
        hit = obstacles.first_contact(p, d, atom["m"], radius)
        if hit is not None:
            travel, edges = hit
            centre = p + travel * d
            return {"valid_start": True, "collision": True, "first_collision_atom": index,
                    "contact": {"travel_in_atom_m": travel, "centre_xz": centre.tolist(),
                                "faces": [int(f) for f in obstacles.face[edges]],
                                "points": obstacles.contact_points(edges, centre).tolist()}}
        p = p + atom["m"] * d
    return {"valid_start": True, "collision": False, "first_collision_atom": None, "contact": None}
