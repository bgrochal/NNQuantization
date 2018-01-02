import math

import numpy as np


def resolve_fixed_point(target_bits, min_value, max_value):
    """
    Returns the following properties describing the fixed-point representation of a dataset to quantize:
      - number of positions to shift each value from the dataset (if the maximal value is in a 0.[fractional part] form);
      - bits width of the fractional part;
      - flag indicating whether values from the dataset require an extra sign bit.
    """
    # Shift the max_value to a form 1.[fractional part] if it is in a 0.[fractional part] form.
    shift_positions = 0
    max_value_abs = math.fabs(max_value)

    while max_value_abs < 1.0:
        max_value_abs *= 10
        shift_positions += 1

    # Check whether the sign bit is needed.
    has_sign = True if min_value < 0 else False

    # Calculate widths of both integer and fractional parts.
    fractional_part, integer_part = math.modf(max_value_abs)
    integer_part_width = math.floor(math.log(integer_part, 2)) + 1
    fractional_part_width = target_bits - integer_part_width - int(has_sign)

    return shift_positions, fractional_part_width, has_sign


def quantize(data, target_bits):
    """
    Returns a quantized dataset consisting of integer values distributed between 0 and (2 ** target_bits) - 1 (both
    inclusive) representing the fixed-point forms of values composing the dataset raised to the power of 2 **
    [fractional part bit width] (i.e. left-shifted by [fractional part bit width] positions).
    """

    def round_value(value):
        rounded_value = int(round(value))
        return rounded_value if rounded_value < quantization_threshold else quantization_threshold - 1

    def represent_binary(value):
        if has_sign:
            bit_sign = '1' if value < 0 else '0'
            return bit_sign + np.binary_repr(abs(value), width=target_bits - 1)
        return np.binary_repr(value, width=target_bits)

    min_value = data.min()
    max_value = data.max()
    shift_positions, fractional_part_width, has_sign = resolve_fixed_point(target_bits, min_value, max_value)

    shift_value = 10 ** shift_positions
    fractional_multiplier = 2 ** fractional_part_width
    quantization_threshold = (1 << (target_bits - int(has_sign)))

    return np.array([represent_binary(round_value(value * shift_value * fractional_multiplier)) for value in
                     data]), shift_positions, fractional_part_width, has_sign


def dequantize(data, shift_positions, fractional_part_width, has_sign):
    """
    Returns a dequantized dataset consisting of fixed-point values corresponding to the rounded values of input dataset.
    """

    def represent_decimal(value):
        if has_sign:
            sign_value = -1 if value[0] == '1' else 1
            return sign_value * int(value[1:], 2)
        return int(value, 2)

    shift_value = 10 ** -shift_positions
    fractional_multiplier = 2 ** -fractional_part_width
    return np.array([represent_decimal(value) * fractional_multiplier * shift_value for value in data]).astype(np.float32)
