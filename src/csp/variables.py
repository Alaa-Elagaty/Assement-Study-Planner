"""Study-session CSP variable construction."""
def build_variables(allocation):
    """Create one variable per required session, ordered assignment then session."""
    variables = []
    for assignment_index, count in enumerate(allocation, start=1):
        if int(count) < 0:
            raise ValueError("allocation counts must be non-negative")
        variables.extend(f"A{assignment_index}_S{session}" for session in range(1, int(count) + 1))
    return variables
