"""ms-swift plugin: BrowserEnv (gym Env over the Mac env service), BrowserScheduler (one record per turn), ActionFormatORM. Module paths verified against ms-swift 4.5.2."""

import asyncio
import copy
import functools
import logging
import os
import uuid

from swift.infer_engine.protocol import RolloutOutput
from swift.rewards.orm import ORM, orms
from swift.rollout.gym_env import Env, envs
from swift.rollout.multi_turn import GYMScheduler, multi_turns

from cua.prompt.format import SYSTEM_PROMPT
from cua.rl.observation import build_observation
from cua.rl.trainer.core import (
    EpisodeRecorder,
    action_format_score,
    blank_png_b64,
    env_client_for,
    episode_records,
    observation_to_user_message,
    require_reward,
)

logger = logging.getLogger(__name__)

RL_ENV_URL = os.environ.get("RL_ENV_URL", "http://127.0.0.1:8900")
RL_ENV_TOKEN = os.environ.get("RL_ENV_TOKEN", "")
RL_RUN_DIR = os.environ.get("RL_RUN_DIR", "/workspace/rl_runs/current")
RL_RUN_ID = os.environ.get("RL_RUN_ID", "grpo")


class BrowserEnv(Env):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.client = env_client_for(RL_ENV_URL, token=RL_ENV_TOKEN)
        self.recorder = EpisodeRecorder(RL_RUN_DIR)
        self.episode_id = None
        self.trajectory_uuid = uuid.uuid4().hex
        self.turn = 0
        self.last_image_path = None
        self.observation = None
        self.close_reason = "external"
        self.close_reward = None
        self.env_error = None
        self.reward_missing = False

    async def _call(self, fn, *args, **kwargs):
        # The env round-trips (slot waits, the verifier on a terminal step) must not block the rollout server's loop, or every concurrent episode stalls with them.
        return await asyncio.get_running_loop().run_in_executor(None, functools.partial(fn, *args, **kwargs))

    async def reset(self, config):
        env_config = (config.data_dict or {}).get("env_config", {})
        try:
            result = await self._call(
                self.client.reset,
                env_config["task_id"],
                int(env_config.get("sample_idx", 0)),
                RL_RUN_ID,
                int(env_config.get("iteration", 0)),
                env_config.get("policy_tag", ""),
            )
        except Exception as e:
            self.env_error = str(e)[:300]
            result = {
                "episode_id": None,
                "observation": build_observation("(env unavailable)", [], blank_png_b64((1280, 720)), 0),
            }
        self.episode_id = result["episode_id"]
        self.observation = result["observation"]
        self.last_image_path = self.recorder.save_png(self.trajectory_uuid, 0, self.observation["image_b64"])
        return self.observation["text"], {"episode_id": self.episode_id, "env_error": self.env_error}, SYSTEM_PROMPT

    async def step(self, action):
        if self.env_error:
            return "", 0.0, True, {"env_error": self.env_error}
        response_text = action[-1]["content"] if action and action[-1]["role"] == "assistant" else ""
        try:
            result = await self._call(self.client.step, self.episode_id, response_text, seq=self.turn)
        except Exception as e:
            # Record it on the env too: an unlabelled transport blip becomes a 0.0 that outranks a
            # genuine failure inside an all-fail group, training toward whatever preceded the blip.
            self.env_error = str(e)[:300]
            return "", 0.0, True, {"env_error": self.env_error}
        self.turn += 1
        observation = result.get("observation")
        if observation is not None:
            self.observation = observation
            self.last_image_path = self.recorder.save_png(self.trajectory_uuid, self.turn, observation["image_b64"])
        reward, done = result.get("reward"), bool(result.get("done"))
        # ms-swift sums step rewards into total_reward, so a verdictless terminal must be flagged here and refused by the scheduler.
        if done and reward is None:
            self.reward_missing = True
        return (
            (observation or {}).get("text", ""),
            float(reward) if reward is not None else 0.0,
            done,
            result.get("info", {}),
        )

    async def close(self):
        if self.episode_id is not None:
            try:
                self.close_reward = (await self._call(self.client.close, self.episode_id, self.close_reason)).get(
                    "reward"
                )
            except Exception:
                logger.exception(f"close failed for episode {self.episode_id}; its reward stays missing")


class BrowserScheduler(GYMScheduler):
    """Each turn is a fresh single-image prompt; run() returns one RolloutOutput per turn carrying the episode reward (spec D2)."""

    def _fresh_prompt(self, env):
        return [{"role": "system", "content": SYSTEM_PROMPT}, observation_to_user_message(env.observation)], [
            env.last_image_path
        ]

    async def run(self, infer_request, request_config, **kwargs):
        await self.on_trajectory_start([infer_request])
        env = self._envs[infer_request.uuid]
        infer_request.messages, infer_request.images = self._fresh_prompt(env)
        turns, turn, turn_result = [], 1, {"rollout_infos": {}}
        while True:
            prompt = copy.deepcopy(infer_request.messages)
            image = infer_request.images[0] if infer_request.images else None
            response = await self.infer_engine.infer_async(infer_request, request_config, **kwargs)
            choice = response.choices[0]
            infer_request.messages.append({"role": "assistant", "content": choice.message.content})
            turn_result = await self.on_turn_end(infer_request, choice, turn)
            turns.append(
                {
                    "response": response,
                    "user": prompt[-1],
                    "assistant": choice.message.content,
                    "token_ids": list(choice.token_ids or []),
                    "image": image,
                }
            )
            if turn_result.get("done"):
                break
            if self.check_finished(infer_request, choice, turn):
                # The env only sees a close for a trainer-side cut, so name it: the reward treats max_turns like its own step cap.
                env.close_reason = "length" if choice.finish_reason == "length" else "max_turns"
                break
            infer_request.messages, infer_request.images = self._fresh_prompt(env)
            turn += 1
        await self._close_and_remove(infer_request.uuid)
        # A trainer-side cut (max_turns / length) is scored by the service at close, not by any step, so that reward must be read back here.
        if turn_result.get("done"):
            total_reward = None if env.reward_missing else (turn_result.get("rollout_infos") or {}).get("total_reward")
        else:
            total_reward = env.close_reward
        if env.env_error is not None and total_reward is None:
            # Env unavailability is the environment's fault, not a verdictless episode; it stays 0.0 but is labeled for the analysis.
            total_reward = 0.0
        total_reward = require_reward(total_reward, infer_request.uuid)
        records = episode_records(SYSTEM_PROMPT, turns, total_reward, infer_request.uuid, env_error=env.env_error)
        return [
            RolloutOutput(
                response=t["response"],
                messages=r["messages"],
                response_token_ids=[t["token_ids"]] if t["token_ids"] else None,
                rollout_infos=r["rollout_infos"],
            )
            for t, r in zip(turns, records, strict=False)
        ]


class ActionFormatORM(ORM):
    def __call__(self, completions, **kwargs):
        return [action_format_score(c if isinstance(c, str) else str(c)) for c in completions]


envs["browser_env"] = BrowserEnv
multi_turns["browser_scheduler"] = BrowserScheduler
orms["action_format"] = ActionFormatORM
