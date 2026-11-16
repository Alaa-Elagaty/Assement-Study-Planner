import unittest
from src.csp.domains import TimeSlot, build_domains
from src.csp.constraints import compatible
from src.csp.backtracking import solve_bt
from src.csp.forward_checking import solve_fc
from src.csp.ac3 import revise, ac3, solve_ac3

class CSPTests(unittest.TestCase):
    def setUp(self): self.slots=[TimeSlot(1,1),TimeSlot(1,2),TimeSlot(2,1)]
    def test_deadline_domains(self):
        d=build_domains([1,1],self.slots,[1,2]);self.assertEqual(len(d["A1_S1"]),2);self.assertEqual(len(d["A2_S1"]),3)
    def test_no_overlap(self):
        a=build_domains([1,1],self.slots,[2,2]);self.assertFalse(compatible("A1_S1",self.slots[0],"A2_S1",self.slots[0]))
    def test_ordering(self):
        self.assertTrue(compatible("A1_S1",self.slots[0],"A1_S2",self.slots[1]))
        self.assertFalse(compatible("A1_S1",self.slots[1],"A1_S2",self.slots[0]))
    def test_bt_solution(self): self.assertTrue(solve_bt(build_domains([1,1],self.slots,[2,2]))["success"])
    def test_fc_prunes_and_solves(self):
        domains=build_domains([2],self.slots,[2]);original={k:list(v) for k,v in domains.items()}
        r=solve_fc(domains);self.assertTrue(r["success"]);self.assertGreater(r["stats"]["domain_values_removed"],0)
        self.assertEqual(domains,original)  # all pruning was on a reversible working copy
    def test_ac3_revision(self):
        d=build_domains([2],self.slots,[2]);d["A1_S1"]=[self.slots[1]];d["A1_S2"]=[self.slots[0],self.slots[2]]
        self.assertTrue(revise(d,"A1_S2","A1_S1"));self.assertEqual(d["A1_S2"],[self.slots[2]])
    def test_ac3_solver(self): self.assertTrue(solve_ac3(build_domains([2,1],self.slots,[2,2]))["success"])
    def test_impossible_schedule(self):
        d=build_domains([1,1], [TimeSlot(1,1)], [1,1])
        for solver in (solve_bt,solve_fc,solve_ac3): self.assertFalse(solver(d)["success"])
    def test_empty_domain_failure(self):
        d=build_domains([2],[TimeSlot(1,1)],[1]);self.assertFalse(solve_ac3(d)["success"])

if __name__=="__main__": unittest.main()
