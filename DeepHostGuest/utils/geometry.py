"""Small 3D geometry helpers used by the structural-augmentation pipeline.

These four routines are dependency-free (NumPy only) re-implementations of the
helpers that the augmentation code originally imported from the internal
``sugar`` package.  They are kept here so that the released repository runs
without any private dependency:

* :func:`cal_rotation_matrix` -- axis/angle to 3x3 rotation matrix (Rodrigues).
* :func:`rotation_around_axis` -- apply a rotation matrix to coordinates.
* :func:`translation` -- apply a translation vector to coordinates.
* :func:`norm_vector` -- normalise a vector to unit length.

The numerical behaviour is identical to the original implementation, so
augmented structures generated with a fixed ``seed`` are bit-for-bit
reproducible across versions.
"""

import numpy as np

__all__ = [
    "cal_rotation_matrix",
    "rotation_around_axis",
    "translation",
    "norm_vector",
]


def translation(position, move):
    """Translate ``position`` by ``move``.

    Parameters
    ----------
    position : array_like
        Coordinate array of shape ``(N, 3)``.
    move : array_like
        Translation vector of shape ``(1, 3)`` or ``(3,)``.

    Returns
    -------
    numpy.ndarray
        Translated coordinates.
    """
    return np.add(np.array(position), np.array(move))


def cal_rotation_matrix(axis, angle):
    """Rotation matrix around ``axis`` by ``angle`` radians (Rodrigues' formula).

    Parameters
    ----------
    axis : array_like
        Rotation axis; normalised internally, so it does not need to be a unit
        vector.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` rotation matrix.
    """
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    one_minus_cos = 1.0 - cos_a
    x, y, z = axis
    rot_matrix = np.array([
        [x * x * one_minus_cos + cos_a, x * y * one_minus_cos - z * sin_a, x * z * one_minus_cos + y * sin_a],
        [x * y * one_minus_cos + z * sin_a, y * y * one_minus_cos + cos_a, y * z * one_minus_cos - x * sin_a],
        [x * z * one_minus_cos - y * sin_a, y * z * one_minus_cos + x * sin_a, z * z * one_minus_cos + cos_a],
    ])
    return rot_matrix.reshape((3, 3))


def rotation_around_axis(position, rot_mat):
    """Apply the rotation matrix ``rot_mat`` to ``position``.

    ``position`` must be given as ``(3, N)`` (i.e. transposed coordinates); the
    rotation is applied about the origin, which is why the augmentation code
    rotates the host and guest toy structures with the *same* matrix.
    """
    return np.dot(rot_mat, position)


def norm_vector(vector):
    """Return ``vector`` normalised to unit length."""
    return np.array(vector) / np.linalg.norm(vector)
