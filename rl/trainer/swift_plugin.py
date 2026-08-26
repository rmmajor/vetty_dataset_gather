"""ms-swift plugin: BrowserEnv (gym Env over the Mac env service), BrowserScheduler (one record per turn), ActionFormatORM. VALIDATE-ON-BOX: ms-swift 4.5.x module paths and scheduler hook names."""
import copy
import os
import uuid

from swift.llm.infer.protocol import RolloutOutput
from swift.plugin.orm import ORM, orms
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
        env_config = config.data_dict.get("env_config", {})
        result = self.client.reset(env_config["task_id"], int(env_config.get("sample_idx", 0)), RL_RUN_ID,
                                   int(env_config.get("iteration", 0)), env_config.get("policy_tag", ""))
        self.episode_id = result["episode_id"]
        self.observation = result["observation"]
        self.last_image_path = self.recorder.save_png(self.trajectory_uuid, 0, self.observation["image_b64"])
        return self.observation["text"], {"episode_id": self.episode_id}, SYSTEM_PROMPT

    async def step(self, messages):
        response_text = messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else ""
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
    """Each turn is a fresh single-image prompt; outputs are one RolloutOutput per turn carrying the episode reward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turns_by_uuid = {}

    def _fresh_prompt(self, env):
        user = observation_to_user_message(env.observation)
        return [{"role": "system", "content": SYSTEM_PROMPT}, user], [env.last_image_path]

    async def on_trajectory_start(self, infer_request, request_config, **kwargs):
        result = await super().on_trajectory_start(infer_request, request_config, **kwargs)
        env = kwargs.get("env")
        if env is not None:
            infer_request.messages, infer_request.images = self._fresh_prompt(env)
        self.turns_by_uuid[infer_request.uuid] = []
        return result

    def step(self, infer_request, response_choice, current_turn, **kwargs):
        env = kwargs.get("env")
        self.turns_by_uuid.setdefault(infer_request.uuid, []).append({
            "user": copy.deepcopy(infer_request.messages[-1]), "assistant": response_choice.message.content,
            "token_ids": getattr(response_choice, "token_ids", None), "image": infer_request.images[0] if infer_request.images else None,
        })
        if env is not None:
            infer_request.messages, infer_request.images = self._fresh_prompt(env)
        return infer_request

    def build_outputs(self, infer_request, total_reward, **kwargs):
        turns = self.turns_by_uuid.pop(infer_request.uuid, [])
        return [RolloutOutput(response=None, messages=r["messages"], response_token_ids=[r["response_token_ids"]] if r["response_token_ids"] else None,
                              rollout_infos=r["rollout_infos"])
                for r in episode_records(SYSTEM_PROMPT, turns, total_reward, infer_request.uuid)]


class ActionFormatORM(ORM):
    def __call__(self, completions, **kwargs):
        return [action_format_score(c if isinstance(c, str) else str(c)) for c in completions]


envs["browser_env"] = BrowserEnv
multi_turns["browser_scheduler"] = BrowserScheduler
orms["action_format"] = ActionFormatORM
