import unittest
from src.integration.effort_model import adjust_features_for_effort, FEATURES
from src.integration.planner import StudyPlanner
from src.csp.domains import TimeSlot

PROFILE=dict(zip(FEATURES,[47,12.5,1,78,78,1200,15,60]))
class MockPredictor:
    def __call__(self, features):
        # Deterministic test model: +1 mark per 100 simulated clicks.
        return {"predicted_mark": min(100, features["previous_tma_mean"] + (features["previous_clicks"]-1200)/100)}

class IntegrationTests(unittest.TestCase):
    def test_effort_copy_and_clicks(self):
        adjusted=adjust_features_for_effort(PROFILE,2,100)
        self.assertEqual(adjusted["previous_clicks"],1400);self.assertEqual(PROFILE["previous_clicks"],1200)
    def test_effort_rejects_wrong_features(self):
        with self.assertRaises(ValueError): adjust_features_for_effort({"date":1},1)
    def test_planner_mock_predictor(self):
        assignments=[{"name":"A1","target_mark":80,"deadline_week":2,"features":PROFILE}]
        result=StudyPlanner(MockPredictor()).plan(assignments,[TimeSlot(1,1),TimeSlot(1,2),TimeSlot(2,1)])
        self.assertTrue(result["success"]);self.assertGreaterEqual(result["predicted_marks"][0],80);self.assertEqual(len(result["timetable"]),result["allocation"][0])
    def test_search_continues_after_csp_rejects_candidate(self):
        # The mock is deliberately non-monotonic, exposing two goal allocations.
        # A custom CSP solver is injected by replacing the selected module-level solver.
        import src.integration.planner as planner_module
        original=planner_module.solve_ac3; calls=[]
        def reject_first(domains):
            calls.append(tuple(domains))
            if len(calls)==1: return {"success":False,"assignment":None,"stats":{"assignments_attempted":0,"backtracks":1,"domain_values_removed":0,"constraint_checks":0,"execution_time_seconds":0,"solution_found":False}}
            return original(domains)
        planner_module.solve_ac3=reject_first
        try:
            class TwoStep:
                def __call__(self,f):
                    # Feature clicks encode session count. At 1 session this profile reaches target.
                    n=(f["previous_clicks"]-1200)//100
                    return {"predicted_mark":90 if n>=1 else 0}
            ass=[{"name":"A","target_mark":80,"deadline_week":2,"features":PROFILE}]
            res=StudyPlanner(TwoStep()).plan(ass,[TimeSlot(1,1),TimeSlot(2,1)])
            self.assertTrue(res["success"]);self.assertGreaterEqual(len(calls),2);self.assertGreaterEqual(res["rejected_goal_count"],1)
        finally: planner_module.solve_ac3=original
    def test_real_predictor_when_models_available(self):
        try:
            from src.nn.predictor import predict_mark
        except Exception as exc:
            self.skipTest(f"Real model dependencies/files unavailable: {exc}")
        result=predict_mark(PROFILE)
        self.assertIn("predicted_mark",result)

if __name__=="__main__": unittest.main()
