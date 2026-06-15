# src/meshing_utils/geometry/vectors.py
"""Pure 3-tuple vector math — no numpy dependency."""

Vec3 = tuple[float, float, float]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


def dot(a: Vec3, b: Vec3) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def norm(v: Vec3) -> float:
    return (v[0]**2 + v[1]**2 + v[2]**2) ** 0.5
