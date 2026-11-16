import unittest
from src.search.state import initial_state, session_cost
from src.search.successor import generate_successors
from src.search.goal_test import is_goal
from src.search.algorithms import uniform_cost_search, greedy_best_first_search

class SearchTests(unittest.TestCase):
    def test_successors(self): self.assertEqual(set(generate_successors((0,0),(1,2))),{(1,0),(0,1)})
    def test_initial_and_cost(self): self.assertEqual(initial_state(3),(0,0,0));self.assertEqual(session_cost((2,1,3)),6)
    def test_goal(self): self.assertTrue(is_goal([60,70],[55,70]));self.assertFalse(is_goal([54,70],[55,70]))
    def test_ucs_minimum_cost(self):
        def evaluate(s): return (50+20*s[0],50+10*s[1])
        r=uniform_cost_search([3,3],[70,60],evaluate)
        self.assertTrue(r.found);self.assertEqual(r.allocation,(1,1));self.assertEqual(r.stats["total_sessions"],2)
    def test_greedy_goal(self):
        def evaluate(s): return (50+20*s[0],50+10*s[1])
        r=greedy_best_first_search([3,3],[70,60],evaluate)
        self.assertTrue(r.found);self.assertTrue(all(x>=t for x,t in zip(r.predicted_marks,[70,60])))
    def test_limits_terminate(self):
        r=uniform_cost_search([1],[101],lambda s:[0])
        self.assertFalse(r.found);self.assertEqual(r.stats["explored_states"],2)
    def test_search_continues_after_rejected_goal(self):
        seen=[]
        def evaluate(s): return [100 if sum(s)>=1 else 0, 100]
        def accept(s,m):
            seen.append(s)
            return s==(0,1)
        r=uniform_cost_search([1,1],[90,90],evaluate,accept)
        self.assertTrue(r.found);self.assertEqual(r.allocation,(0,1));self.assertGreaterEqual(len(seen),2)
    def test_bad_dimensions(self):
        with self.assertRaises(ValueError): list(generate_successors((0,),(1,2)))

if __name__=="__main__": unittest.main()
