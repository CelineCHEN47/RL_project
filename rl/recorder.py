"""Per-round training trace recorder.

Captures two artifacts under ``<save_dir>/``:

1. ``training_summary.json`` — one row per round with aggregate stats
   (tag count, tagger id, mean tagger→runner distance, summed reward).
   Always written; small enough to read in full.

2. ``training_traces/round_<N>.json`` — decimated per-frame trace for
   each round: positions, actions, instantaneous rewards, tag events.
   Decimation defaults to every 10 steps (≈200 samples for a 2000-step
   round) to keep each file small (~30 KB).

The recorder is decoupled from HeadlessSimulation: the experiment
loop drives it via ``start_round`` / ``record_step`` / ``end_round``.
"""

import json
import os


class TrainingRecorder:
    def __init__(self, save_dir: str, decimation: int = 10,
                 record_rewards: bool = True):
        self.save_dir = save_dir
        self.decimation = max(1, int(decimation))
        self.record_rewards = record_rewards
        self.traces_dir = os.path.join(save_dir, "training_traces")
        os.makedirs(self.traces_dir, exist_ok=True)
        self.summary: list[dict] = []
        self._current: dict | None = None
        self._round_reward_sum: dict[int, float] = {}
        self._round_dist_sum = 0.0
        self._round_dist_count = 0
        self._tag_count = 0

    def start_round(self, round_num: int, sim):
        self._current = {
            "round": round_num,
            "decimation": self.decimation,
            "agents": [{"id": a.entity_id} for a in sim.agents],
            "frames": [],
            "tag_events": [],
        }
        self._round_reward_sum = {a.entity_id: 0.0 for a in sim.agents}
        self._round_dist_sum = 0.0
        self._round_dist_count = 0
        self._tag_count = 0

    def record_step(self, step: int, sim, tag_event: dict | None):
        if self._current is None:
            return

        rewards = getattr(sim, "last_rewards", {})

        # Accumulate round-level reward totals (all steps)
        for aid, r in rewards.items():
            if aid in self._round_reward_sum:
                self._round_reward_sum[aid] += r

        # Accumulate tagger → nearest runner distance (for summary)
        tagger = next((a for a in sim.agents if a.is_tagger), None)
        if tagger is not None:
            min_d = float("inf")
            for a in sim.agents:
                if a is tagger or a.is_eliminated:
                    continue
                d = tagger.distance_to(a)
                if d < min_d:
                    min_d = d
            if min_d < float("inf"):
                self._round_dist_sum += min_d
                self._round_dist_count += 1

        # Record tag events (all of them — they're sparse)
        if tag_event:
            self._tag_count += 1
            self._current["tag_events"].append({
                "t": step,
                "tagger_id": tag_event["tagger_id"],
                "tagged_id": tag_event["tagged_id"],
                "pos": [round(tag_event["tagged_pos"][0], 1),
                        round(tag_event["tagged_pos"][1], 1)],
            })

        # Decimated per-frame snapshot
        if step % self.decimation == 0:
            agents = sim.agents
            tagger_id = next((a.entity_id for a in agents if a.is_tagger), -1)
            frame = {
                "t": step,
                "tagger": tagger_id,
                "pos": [[round(a.x, 1), round(a.y, 1)] for a in agents],
                "act": [int(a.last_action) for a in agents],
            }
            if self.record_rewards:
                frame["rew"] = [round(rewards.get(a.entity_id, 0.0), 3)
                                for a in agents]
            self._current["frames"].append(frame)

    def end_round(self, sim, steps_per_round: int):
        if self._current is None:
            return

        cur = self._current
        cur["steps_per_round"] = steps_per_round

        # Write per-round trace file (compact JSON, no indentation)
        path = os.path.join(self.traces_dir, f"round_{cur['round']}.json")
        with open(path, "w") as f:
            json.dump(cur, f, separators=(",", ":"))

        # Append summary row
        mean_dist = (self._round_dist_sum / self._round_dist_count
                     if self._round_dist_count > 0 else None)
        tagger_id = next((a.entity_id for a in sim.agents if a.is_tagger), -1)
        tagger_reward = self._round_reward_sum.get(tagger_id, 0.0)
        runner_reward_total = sum(
            r for aid, r in self._round_reward_sum.items() if aid != tagger_id
        )
        num_runners = max(1, sum(1 for a in sim.agents if a.entity_id != tagger_id))

        self.summary.append({
            "round": cur["round"],
            "tag_count": self._tag_count,
            "tagger_id": tagger_id,
            "mean_tagger_to_nearest_runner": (round(mean_dist, 2)
                                              if mean_dist is not None else None),
            "tagger_total_reward": round(tagger_reward, 3),
            "runner_mean_total_reward": round(runner_reward_total / num_runners, 3),
            "tag_event_steps": [ev["t"] for ev in cur["tag_events"]],
        })
        self._current = None

    def finalize(self):
        path = os.path.join(self.save_dir, "training_summary.json")
        with open(path, "w") as f:
            json.dump(self.summary, f, indent=2)
        return path
