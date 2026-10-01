from nano_mle.search import Greedy, MCGS, MCTS, update_stats


def candidate(id, parent="root", score=1):
    return {"id": id, "parent_id": parent, "status": "ok", "score": score}


def test_greedy_ignores_failed_candidates():
    good = candidate("good")
    bad = {**candidate("bad", score=100), "status": "failed"}
    assert Greedy().select([bad, good], {}, 0).parent_id == "good"


def test_sibling_credit_does_not_flow_through_references():
    candidates = [candidate("parent"), candidate("reference"), candidate("a", "parent", 2),
                  candidate("b", "parent", 3)]
    candidates[-1]["reference_ids"] = ["reference"]
    stats = update_stats({}, candidates, candidates[-2:], baseline=1)
    assert stats["parent"] == {"visits": 2, "reward_sum": 4}
    assert stats["root"] == {"visits": 2, "reward_sum": 4}
    assert "reference" not in stats
    assert stats["a"]["reward_sum"] == stats["b"]["reward_sum"] == 2


def test_mcts_can_return_to_root_and_mcgs_records_references():
    candidates = [candidate("a"), candidate("b", score=2)]
    assert MCTS().select(candidates[:1], {"root": {"visits": 1}}, 1).parent_id == "root"
    selection = MCGS(max_steps=10).select(candidates, {}, 0)
    assert selection.parent_id == "a"
    assert selection.reference_ids == ["b"]
