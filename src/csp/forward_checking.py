"""Backtracking with forward checking and reversible domain pruning."""
from time import perf_counter
from .constraints import compatible, neighbors

def solve_fc(domains):
    """Solve CSP, pruning incompatible values from unassigned neighbors."""
    started=perf_counter(); variables=list(domains); current={v:list(d) for v,d in domains.items()}; assignment={}
    stats={"assignments_attempted":0,"backtracks":0,"domain_values_removed":0,"constraint_checks":0}
    def visit():
        if len(assignment)==len(variables): return True
        var=next(v for v in variables if v not in assignment)
        for value in list(current[var]):
            stats["assignments_attempted"]+=1
            if any(not compatible(var,value,other,oval) for other,oval in assignment.items()):
                stats["constraint_checks"]+=len(assignment); continue
            assignment[var]=value; changes=[]; failed=False
            for neighbor in neighbors(var,variables):
                if neighbor in assignment: continue
                old=list(current[neighbor]); new=[candidate for candidate in old if compatible(var,value,neighbor,candidate)]
                stats["constraint_checks"]+=len(old); stats["domain_values_removed"]+=len(old)-len(new)
                if len(new)!=len(old): changes.append((neighbor,old));current[neighbor]=new
                if not new: failed=True;break
            if not failed and visit(): return True
            for neighbor,old in reversed(changes): current[neighbor]=old
            del assignment[var]
        stats["backtracks"]+=1
        return False
    found=visit();stats.update({"execution_time_seconds":perf_counter()-started,"solution_found":found})
    return {"success":found,"assignment":dict(assignment) if found else None,"stats":stats,"remaining_domains":current}
