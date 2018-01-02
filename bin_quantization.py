import numpy as np


def resolve_integer_type(target_bits):
    """
    Returns appropriate numpy's uint type of given width based on the number of target_bits.
    """
    types_mapping = {1: np.bool_, 8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    return types_mapping[min({key: value for key, value in types_mapping.items() if key >= target_bits})]


def quantize(data, target_bits):
    """
    Returns the number of a bin containing each of the values defined in data. All of the bins, except the last one, are
    left-closed and right-opened (the last bin is also right-closed).
    """
    min_value = data.min()
    max_value = data.max()
    bin_width = (max_value - min_value) / (1 << target_bits)

    data_type = resolve_integer_type(target_bits)
    greatest_bin_number = (1 << target_bits) - 1
    return np.array([np.binary_repr(((value - min_value) // bin_width).astype(data_type) if value < max_value else greatest_bin_number, width=target_bits) for value in data]), min_value, max_value


def dequantize(data, source_bits, min_value, max_value):
    """
    Returns centers of all bins being the values defined in data.
    """
    bin_width = (max_value - min_value) / (1 << source_bits)
    return np.array([min_value + (int(value, 2) + 0.5) * bin_width for value in data]).astype(np.float32)
