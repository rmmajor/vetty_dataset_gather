"""ms-swift plugin: BrowserEnv (gym Env over the Mac env service), BrowserScheduler (one record per turn), ActionFormatORM. Module paths verified against ms-swift 4.5.2."""
import copy
import os
import uuid

from swift.infer_engine.protocol import RolloutOutput
from swift.rewards.orm import ORM, orms
from swift.rollout.gym_env import Env, envs
from swift.rollout.multi_turn import GYMScheduler, multi_turns

from rl.trainer.core import EpisodeRecorder, action_format_score, env_client_for, episode_records, observation_to_user_message
from train_qwen import SYSTEM_PROMPT

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

    async def reset(self, config):
        env_config = (config.data_dict or {}).get("env_config", {})
        result = self.client.reset(env_config["task_id"], int(env_config.get("sample_idx", 0)), RL_RUN_ID,
                                   int(env_config.get("iteration", 0)), env_config.get("policy_tag", ""))
        self.episode_id = result["episode_id"]
        self.observation = result["observation"]
        self.last_image_path = self.recorder.save_png(self.trajectory_uuid, 0, self.observation["image_b64"])
        return self.observation["text"], {"episode_id": self.episode_id}, SYSTEM_PROMPT

    async def step(self, action):
        response_text = action[-1]["content"] if action and action[-1]["role"] == "assistant" else ""
        try:
            result = self.client.step(self.episode_id, response_text)
        except Exception as e:
            return "", 0.0, True, {"env_error": str(e)[:300]}
        self.turn += 1
        observation = result.get("observation")
        if observation is not None:
            self.observation = observation
            self.last_image_path = self.recorder.save_png(self.trajectory_uuid, self.turn, observation["image_b64"])
        reward = result.get("reward")
        return (observation or {}).get("text", ""), float(reward or 0.0), bool(result.get("done")), result.get("info", {})

    async def close(self):
        if self.episode_id is not None:
            try:
                self.client.close(self.episode_id, "external")
            except Exception:
                pass


class BrowserScheduler(GYMScheduler):
    """Each turn is a fresh single-image prompt; run() returns one RolloutOutput per turn carrying the episode reward (spec D2)."""

    def _fresh_prompt(self, env):
        return [{"role": "system", "content": SYSTEM_PROMPT}, observation_to_user_message(env.observation)], [env.last_image_path]

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
            turns.append({"response": response, "user": prompt[-1], "assistant": choice.message.content,
                          "token_ids": list(choice.token_ids or []), "image": image})
            if turn_result.get("done") or self.check_finished(infer_request, choice, turn):
                break
            infer_request.messages, infer_request.images = self._fresh_prompt(env)
            turn += 1
        await self._close_and_remove(infer_request.uuid)
        total_reward = float((turn_result.get("rollout_infos") or {}).get("total_reward", 0.0))
        records = episode_records(SYSTEM_PROMPT, turns, total_reward, infer_request.uuid)
        return [RolloutOutput(response=t["response"], messages=r["messages"], response_token_ids=[t["token_ids"]] if t["token_ids"] else None,
                              rollout_infos=r["rollout_infos"]) for t, r in zip(turns, records)]


class ActionFormatORM(ORM):
    def __call__(self, completions, **kwargs):
        return [action_format_score(c if isinstance(c, str) else str(c)) for c in completions]


envs["browser_env"] = BrowserEnv
multi_turns["browser_scheduler"] = BrowserScheduler
orms["action_format"] = ActionFormatORM
