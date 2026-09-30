from copy import deepcopy
import pytest
from actor_repair import (torch, TD3, SCENARIOS, VARIANTS, prepare, actor_step,
                          state_batch, select_variant)


def test_balanced_batch_and_repeated_schedule():
    data = {s: {"observations": torch.stack((torch.full((150,), float(i)),
            torch.arange(150).float(), torch.arange(150).float()+100), dim=1)} for i, s in enumerate(SCENARIOS)}
    a, b = torch.Generator().manual_seed(7100), torch.Generator().manual_seed(7100)
    for _ in range(5):
        first, second = state_batch(data, a), state_batch(data, b)
        assert torch.equal(first, second)
        assert first.shape == (40, 3)
        for i in range(5):
            assert int((first[:, 0] == i).sum()) == 8


@pytest.mark.parametrize("variant", VARIANTS)
def test_frozen_critic_no_base_mutation_and_head_policy(variant):
    base = TD3(3, 6300, 8)
    with torch.no_grad():
        base.actor[-2].bias[:] = torch.tensor([3., -3.])
    before = deepcopy(base.actor.state_dict())
    learner = prepare(base, variant)
    if variant == "reset_head":
        assert torch.equal(torch.from_numpy(learner.act([0., 0., 0.])), torch.zeros(2))
    else:
        assert all(torch.equal(v, before[k]) for k, v in learner.actor.state_dict().items())
    critic = deepcopy(learner.critics.state_dict())
    actor_step(learner, torch.ones((40, 3)))
    assert all(torch.equal(v, critic[k]) for k, v in learner.critics.state_dict().items())
    assert all(torch.equal(v, before[k]) for k, v in base.actor.state_dict().items())
    assert all(torch.equal(v, before[k]) for k, v in learner.actor.state_dict().items()
               if not k.startswith("4."))
    assert all(p.requires_grad == name.startswith("4.") for name, p in learner.actor.named_parameters())
    assert learner.updates == 0


def test_reject_unknown_variant():
    with pytest.raises(ValueError):
        prepare(TD3(3, 1, 8), "wrong")


def test_selection_requires_gain_and_gap_reduction_and_stable_tie():
    initial = dict(equal_scenario_q=-10., mean_grid_gap=1.)
    complete = {s: dict(equal_scenario_q=-9., mean_grid_gap=.5) for s in VARIANTS}
    assert select_variant(initial, complete) == "continue"
    complete["reset_head"]["equal_scenario_q"] = -8.
    assert select_variant(initial, complete) == "reset_head"
    complete["reset_head"]["mean_grid_gap"] = .8
    assert select_variant(initial, complete) is None


def test_preexisting_stop_prevents_model_load(tmp_path, monkeypatch):
    import actor_repair
    out = tmp_path / "fit"
    (tmp_path / "STOP").touch()
    monkeypatch.setattr("sys.argv", ["actor_repair.py", "--output", str(out)])
    monkeypatch.setattr(actor_repair, "load_base", lambda: pytest.fail("model load after STOP"))
    actor_repair.main()
    assert actor_repair.read(out / "partial.json")["status"] == "paused"


@pytest.mark.parametrize("variant", VARIANTS)
def test_hidden_weights_with_populated_adam_history_remain_fixed(variant):
    base = TD3(3, 6300, 8)
    for _ in range(3):
        actor_step(base, torch.ones((40, 3)))
    with torch.no_grad():
        base.actor[-2].bias[:] = torch.tensor([3., -3.])
    learner = prepare(base, variant)
    initial = deepcopy(learner.actor.state_dict())
    for _ in range(5):
        actor_step(learner, torch.ones((40, 3)))
    assert all(torch.equal(v, initial[k]) for k, v in learner.actor.state_dict().items()
               if not k.startswith("4."))
    assert any(not torch.equal(v, initial[k]) for k, v in learner.actor.state_dict().items()
               if k.startswith("4."))
