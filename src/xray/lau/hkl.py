import numpy as np


def is_unmixed(h, k, ell):
    """
    Checks if Miller indices are 'unmixed' (all even or all odd).
    Zero is considered even.
    """
    h_even = h % 2 == 0
    k_even = k % 2 == 0
    ell_even = ell % 2 == 0

    all_even = h_even and k_even and ell_even
    all_odd = (not h_even) and (not k_even) and (not ell_even)

    return all_even or all_odd


def find_hkl(x, y, z):
    """
    Finds the smallest unmixed (h, k, l) triple based on the
    ratio x:y:z. Returns standard Python ints.
    """
    coords = np.array([x, y, z])

    non_zero_abs_coords = np.abs(coords[np.abs(coords) > 1e-6])

    if len(non_zero_abs_coords) == 0:
        return (0, 0, 0)

    min_val = np.min(non_zero_abs_coords)

    base_ratio = coords / min_val
    base_integers = np.round(base_ratio).astype(int)

    if np.all(base_integers == 0):
        base_integers = np.round(base_ratio * 10).astype(int)
        if np.all(base_integers == 0):
            return (0, 0, 0)

    h_base, k_base, l_base = base_integers

    for multiplier in range(1, 10):
        h = h_base * multiplier
        k = k_base * multiplier
        ell = l_base * multiplier  # Renamed l to ell

        if is_unmixed(h, k, ell):
            return (int(h), int(k), int(ell))  # Return standard ints

    return (int(h_base), int(k_base), int(l_base))  # Return standard ints
