def step_function(x: float) -> int:
    """
    Step activation function.

    Parameters:
        x (float): Input value.

    Returns:
        int: Output value (0 or 1).
    """
    return 1 if x >= 0 else 0
