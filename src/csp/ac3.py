"""Backtracking with genuine AC-3 arc-consistency propagation."""
from collections import deque
from time import perf_counter
from .constraints import compatible, neighbors

def revise(domains, xi, xj, stats=None):
    """Remove Xi values lacking any compatible support in Xj; return changed."""
    revised=False; kept=[]
    for x in domains[xi]:
        supported=False
        for y in domains[xj]:
            if stats is not None: stats["constraint_checks"]+=1
            if compatible(xi,x,xj,y): supported=True;break
        if supported: kept.append(x)
        else:
            revised=True
            if stats is not None: stats["domain_values_removed"]+=1
    if revised: domains[xi]=kept
    return revised

def ac3(domains, arcs=None, stats=None):
    """Enforce arc consistency with a queue; False indicates domain wipeout."""
    variables=list(domains)
    queue=deque(arcs if arcs is not None else [(xi,xj) for xi in variables for xj in neighbors(xi,variables)])
    while queue:
        xi,xj=queue.popleft()
        if revise(domains,xi,xj,stats):
            if not domains[xi]: return False
            for xk in neighbors(xi,variables):
                if xk!=xj: queue.append((xk,xi))
    return True

def solve_ac3(domains):
    """Backtracking solver that re-establishes arc consistency after each choice."""
    started=perf_counter(); variables=list(domains); base={v:list(d) for v,d in domains.items()}; assignment={}
    stats={"assignments_attempted":0,"backtracks":0,"domain_values_removed":0,"constraint_checks":0}
    def visit(current):
        if len(assignment)==len(variables): return dict(assignment)
        var=next(v for v in variables if v not in assignment)
        for value in current[var]:
            stats["assignments_attempted"]+=1
            if any(not compatible(var,value,other,oval) for other,oval in assignment.items()):
                stats["constraint_checks"]+=len(assignment);continue
            assignment[var]=value
            child={v:list(d) for v,d in current.items()}
            child[var]=[value]
            # Propagate the assignment's restrictions to every other variable,
            # then AC-3 continues propagation through the resulting arcs.
            for other in variables:
                if other!=var and other not in assignment:
                    old=len(child[other]);child[other]=[x for x in child[other] if compatible(var,value,other,x)]
                    stats["constraint_checks"]+=old;stats["domain_values_removed"]+=old-len(child[other])
            if all(child[v] for v in variables if v not in assignment) and ac3(child,stats=stats):
                result=visit(child)
                if result is not None:return result
            del assignment[var]
        stats["backtracks"]+=1
        return None
    # AC-3 FIRST: trim impossible slots before any assignment is tried.
    before=sum(len(d) for d in base.values())
    if not ac3(base,stats=stats):
        stats.update({"root_ac3_removed":before-sum(len(d) for d in base.values()),"execution_time_seconds":perf_counter()-started,"solution_found":False})
        return {"success":False,"assignment":None,"stats":stats}
    stats["root_ac3_removed"]=before-sum(len(d) for d in base.values())
    result=visit(base);stats.update({"execution_time_seconds":perf_counter()-started,"solution_found":result is not None})
    return {"success":result is not None,"assignment":result,"stats":stats}
