"""Tests for NAF episodes and PPO training (NAF, Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from openbo.optimizers.naf import (
    BOEpisode,
    FamilyGPHyperparameters,
    NAFConfig,
    _flush_denormal_gradient,
    gae_advantages,
    load_naf,
    log_regret_reward,
    ppo_loss,
    run_episodes,
    train_naf,
)


@dataclass(frozen=True)
class PeakAt08:
    """A 1D family whose maximum is always at x = 0.8; only the output scale varies."""

    scaling: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return self.scaling * (1.0 - 20.0 * (x[:, 0] - 0.8) ** 2)

    @property
    def optimum(self) -> float:
        return self.scaling


TOY_HP = FamilyGPHyperparameters(lengthscale=(0.2,), variance=1.0)
TOY_POOL = [PeakAt08(s) for s in np.linspace(0.9, 1.1, 10)]


def _toy_config(**overrides: object) -> NAFConfig:
    settings: dict[str, object] = dict(
        dim=1,
        budget=3,
        hidden=(32, 32),
        n_local=50,
        n_iterations=2,
        steps_per_iteration=60,
        minibatch_size=20,
        learning_rate=3e-3,
        seed=0,
    )
    settings.update(overrides)
    return NAFConfig(**settings)  # type: ignore[arg-type]


def test_gae_matches_a_hand_computation() -> None:
    rewards = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5])
    values = np.array([0.5, 1.0, 1.5, 0.2, 0.1, 0.0])
    gamma, lam = 0.9, 0.8
    advantages = gae_advantages(rewards, values, budget=3, gamma=gamma, gae_lambda=lam)
    # Episode 1, backwards: the last step does not bootstrap.
    d2 = 3.0 - 1.5
    d1 = 2.0 + gamma * 1.5 - 1.0
    d0 = 1.0 + gamma * 1.0 - 0.5
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1
    np.testing.assert_allclose(advantages[:3], [a0, a1, a2])
    # Episode 2 is independent of episode 1.
    np.testing.assert_allclose(advantages[3:], gae_advantages(rewards[3:], values[3:], 3, gamma, lam))
    with pytest.raises(ValueError, match="whole episodes"):
        gae_advantages(rewards[:5], values[:5], budget=3, gamma=gamma, gae_lambda=lam)


def test_ppo_loss_matches_a_hand_computation() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    log_p = torch.log_softmax(logits, dim=-1)
    old = log_p[0, 1] - 0.05  # ratio exp(0.05), inside the clip range
    loss, stats = ppo_loss(
        logits,
        torch.tensor([1]),
        old[None],
        torch.tensor([2.0]),
        torch.tensor([0.5]),
        torch.tensor([1.5]),
        clip=0.15,
        value_coefficient=1.0,
        entropy_coefficient=0.01,
    )
    # One sample: no advantage normalization.
    entropy = float(-(log_p.exp() * log_p).sum())
    expected = -np.exp(0.05) * 2.0 + 1.0 * (0.5 - 1.5) ** 2 - 0.01 * entropy
    assert float(loss) == pytest.approx(expected, rel=1e-5)
    assert stats["entropy"] == pytest.approx(entropy, rel=1e-5)
    # Outside the clip range the ratio is clipped (for a positive advantage).
    clipped, _ = ppo_loss(
        logits, torch.tensor([1]), (log_p[0, 1] - 0.5)[None], torch.tensor([2.0]),
        torch.tensor([0.0]), torch.tensor([0.0]), 0.15, 1.0, 0.0,
    )
    assert float(clipped) == pytest.approx(-1.15 * 2.0, rel=1e-5)


def test_denormal_gradients_are_flushed_to_zero() -> None:
    """A trained policy's far-apart logits make denormal gradients, which slow the CPU update."""
    tiny = torch.finfo(torch.float32).tiny
    grad = torch.tensor([1e-40, -1e-40, tiny, -2.0 * tiny, 0.5], dtype=torch.float32)
    assert grad[0] != 0.0  # a real float32 denormal
    flushed = _flush_denormal_gradient(grad)
    assert flushed.tolist() == [0.0, 0.0, grad[2].item(), grad[3].item(), 0.5]


def test_episode_rewards_are_negative_log_regret() -> None:
    config = _toy_config()
    episode = BOEpisode(TOY_POOL[0], TOY_HP, config.adaptive_set(), budget=3)
    af = config.new_af()
    rewards = []
    while not episode.done:
        features, logits = episode.observation(af)
        assert features.shape == (config.adaptive_set().size, 5) and logits.shape == (36,)
        rewards.append(episode.step(int(np.argmax(logits))))
    assert episode.t == 3 and len(rewards) == 3
    assert rewards[-1] == pytest.approx(log_regret_reward(episode.regret))
    assert np.all(np.diff(rewards) >= -1e-12)  # simple regret never gets worse
    with pytest.raises(RuntimeError, match="over"):
        episode.observation(af)
    assert log_regret_reward(0.0) == 20.0


def test_training_is_reproducible_and_saves_checkpoints(tmp_path) -> None:
    config = _toy_config()
    first = train_naf(config, TOY_HP, task_pool=TOY_POOL, checkpoint_dir=tmp_path)
    second = train_naf(config, TOY_HP, task_pool=TOY_POOL)
    assert first.mean_return == second.mean_return
    for p1, p2 in zip(first.af.parameters(), second.af.parameters()):
        torch.testing.assert_close(p1, p2)
    assert len(first.mean_return) == 2 and 0 <= first.best_iteration < 2
    for name in ("af.pt", "best_af.pt", "naf_config.json", "training_history.json", "training_state.pt"):
        assert (tmp_path / name).exists()
    loaded, loaded_config, loaded_hp = load_naf(tmp_path, name="best_af")
    assert loaded_config == config and loaded_hp == TOY_HP
    features = torch.randn(4, 9, 5)
    torch.testing.assert_close(loaded(features), first.best_af(features))


def test_resumed_training_matches_an_uninterrupted_run(tmp_path) -> None:
    full = train_naf(_toy_config(n_iterations=4), TOY_HP, task_pool=TOY_POOL)
    train_naf(_toy_config(n_iterations=2), TOY_HP, task_pool=TOY_POOL, checkpoint_dir=tmp_path)
    resumed = train_naf(_toy_config(n_iterations=4), TOY_HP, task_pool=TOY_POOL, checkpoint_dir=tmp_path, resume=True)
    assert resumed.mean_return == full.mean_return
    assert resumed.best_iteration == full.best_iteration
    for p1, p2 in zip(resumed.af.parameters(), full.af.parameters()):
        torch.testing.assert_close(p1, p2)
    with pytest.raises(ValueError, match="Only n_iterations"):
        train_naf(_toy_config(n_iterations=5, learning_rate=1e-2), TOY_HP, task_pool=TOY_POOL, checkpoint_dir=tmp_path, resume=True)


def test_parallel_rollouts_run_every_episode() -> None:
    result = train_naf(_toy_config(n_iterations=1, n_workers=2), TOY_HP, task_pool=TOY_POOL)
    assert len(result.mean_return) == 1 and np.isfinite(result.mean_return[0])


def test_task_sampler_and_validation() -> None:
    def sampler(n: int, seed: int) -> list[PeakAt08]:
        rng = np.random.default_rng(seed)
        return [PeakAt08(float(s)) for s in rng.uniform(0.9, 1.1, n)]

    result = train_naf(_toy_config(n_iterations=1), TOY_HP, task_sampler=sampler)
    assert len(result.mean_return) == 1
    with pytest.raises(ValueError, match="exactly one"):
        train_naf(_toy_config(), TOY_HP)
    with pytest.raises(ValueError, match="multiple of budget"):
        _toy_config(steps_per_iteration=61)


def test_short_training_learns_where_the_family_peaks() -> None:
    """After training, the greedy AF's first pick is near the family's peak at x = 0.8."""
    # 240 steps per iteration (not 60) keeps the gradient noise low enough that every seed learns the peak.
    config = _toy_config(n_iterations=100, steps_per_iteration=240, minibatch_size=60)
    result = train_naf(config, TOY_HP, task_pool=TOY_POOL)

    def first_pick(af: object) -> float:
        # The AF sees the budget T as a feature, so use the training budget.
        episode = BOEpisode(TOY_POOL[0], TOY_HP, config.adaptive_set(), config.budget)
        _, logits = episode.observation(af)  # type: ignore[arg-type]
        assert episode.candidates is not None
        return float(episode.candidates[int(np.argmax(logits)), 0])

    assert abs(first_pick(result.af) - 0.8) < 0.05
    assert abs(first_pick(config.new_af()) - 0.8) > 0.2
    rng = np.random.default_rng(0)
    trained = run_episodes(result.af, TOY_POOL[:3], TOY_HP, config.adaptive_set(), config.budget, rng, greedy=True)
    assert np.max(trained.final_regrets) < 0.01
    assert np.mean(result.mean_final_log_regret[-20:]) < np.mean(result.mean_final_log_regret[:20]) - 1.0
