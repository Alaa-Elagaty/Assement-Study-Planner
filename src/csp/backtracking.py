"""Plain chronological backtracking CSP solver."""
from time import perf_counter
from .constraints import consistent

def solve_bt(domains):
    """Solve using basic backtracking and static variable/domain order."""
    started=perf_counter(); variables=list(domains); assignment={}; stats={"assignments_attempted":0,"backtracks":0,"domain_values_removed":0,"constraint_checks":0}
    def visit(index):
        if index==len(variables): return True
        var=variables[index]
        for value in domains[var]:
            stats["assignments_attempted"]+=1
            ok=True
            for other,other_value in assignment.items():
                stats["constraint_checks"]+=1
                if not consistent(var,value,{other:other_value}): ok=False;break
            if ok:
                assignment[var]=value
                if visit(index+1): return True
                del assignment[var]
        stats["backtracks"]+=1
        return False
    found=visit(0); stats.update({"execution_time_seconds":perf_counter()-started,"solution_found":found})
    return {"success":found,"assignment":dict(assignment) if found else None,"stats":stats}
