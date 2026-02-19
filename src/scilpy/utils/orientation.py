# -*- coding: utf-8 -*-

import re


def validate_voxel_order(axcodes, dimensions=3):
    """
    Validate a set of axis codes.
    Parameters
    ----------
    axcodes : str or tuple or list
        The axis codes to validate (e.g., "LPS", ("R", "A", "S")).
    dimensions : int
        The number of dimensions of the image.
    Returns
    -------
    tuple
        A tuple of validated axis codes.
    Raises
    ------
    ValueError
        If the axis codes are invalid.
    """
    if axcodes is None:
        raise ValueError("Axis codes cannot be None.")

    axcodes = tuple(axcodes)
    
    # We only validate the first 3 spatial dimensions for voxel order
    # even if the image is 4D.
    if len(axcodes) != 3:
        raise ValueError(f"Target axis codes must be of length 3 (spatial).")

    # Check unique are only valid axis codes
    valid_codes = {"L", "R", "A", "P", "S", "I"}
    for code in axcodes:
        if code not in valid_codes:
            raise ValueError(f"Invalid axis code '{code}' in target.")

    # Check no repeated axis codes (LL, RR, etc.)
    if len(set(axcodes)) != 3:
        raise ValueError("Target axis codes must be unique.")

    # Check L/R, A/P, S/I pairs are not both present
    pairs = [("L", "R"), ("A", "P"), ("S", "I")]
    for pair in pairs:
        if pair[0] in axcodes and pair[1] in axcodes:
            raise ValueError(f"Conflicting axis codes '{pair[0]}' and "
                             f"'{pair[1]}' in target.")
    return axcodes


def parse_voxel_order(order_str, dimensions=3):
    """
    Parse the voxel order string into a tuple of axis codes.
    """
    order_str_cleaned = order_str.replace(',', '').replace(' ', '')

    if order_str_cleaned.isalpha():
        if len(order_str_cleaned) > 3:
            raise NotImplementedError("Voxel order longer than 3 is not "
                                      "implemented yet.")
        if len(order_str_cleaned) != 3:
            raise ValueError("Voxel order string must have 3 characters.")
        
        return validate_voxel_order(tuple(order_str_cleaned.upper()), 
                                    dimensions=dimensions)

    if order_str_cleaned.replace('-', '').isdigit():
        numeric_parts = re.findall(r'-?\d', order_str_cleaned)
        
        if len(numeric_parts) > 3:
             raise NotImplementedError("Voxel order longer than 3 is not "
                                       "implemented yet.")

        if len(numeric_parts) != 3:
            raise ValueError("Voxel order string must have 3 numbers.")

        ras_map = {1: 'R', 2: 'A', 3: 'S'}
        flip_map = {'R': 'L', 'A': 'P', 'S': 'I'}

        order = []
        for part in numeric_parts:
            num = int(part)
            axis = ras_map[abs(num)]
            if num < 0:
                axis = flip_map[axis]
            order.append(axis)

        # Check for duplicate axes
        if len(set(order)) != 3:
            # Handle swapped axes from numeric input (e.g., '231')
            axis_vals = [ras_map[abs(int(p))] for p in numeric_parts]
            if len(set(axis_vals)) == 3:
                return validate_voxel_order(tuple(order), dimensions=3)
            else:
                raise ValueError("Invalid numeric voxel order. "
                                 "Axes cannot be repeated.")

        return validate_voxel_order(tuple(order), dimensions=3)
    
    raise ValueError(f"Invalid voxel order format: {order_str}")
