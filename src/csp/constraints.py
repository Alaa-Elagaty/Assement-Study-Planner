"""Binary CSP constraints: all-different and per-assignment chronology."""
def assignment_index(variable: str) -> int:
    return int(variable.split("_")[0][1:])
def session_index(variable: str) -> int:
    return int(variable.split("_")[1][1:])
def are_neighbors(left: str, right: str) -> bool:
    return left != right
def compatible(left_var, left_value, right_var, right_value) -> bool:
    """Check pairwise no-overlap and chronological order constraints."""
    if left_value == right_value:
        return False
    if assignment_index(left_var)==assignment_index(right_var):
        if session_index(left_var) < session_index(right_var):
            return left_value.key < right_value.key
        return right_value.key < left_value.key
    return True
def consistent(variable, value, assignment) -> bool:
    return all(compatible(variable, value, other, other_value) for other, other_value in assignment.items())
def neighbors(variable, variables):
    """Every other variable is a neighbor because all-different is global."""
    return [other for other in variables if other != variable]
