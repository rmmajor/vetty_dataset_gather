"""The student's observation: the exact SFT user turn plus the PNG, and its OpenAI-message rendering."""

from cua.sft.legacy_train_qwen import SYSTEM_PROMPT, build_step_user_text


def build_observation(instruction, action_history, image_png_b64, step_index):
    return {
        "system": SYSTEM_PROMPT,
        "text": build_step_user_text(instruction, action_history),
        "image_b64": image_png_b64,
        "step_index": step_index,
    }


def to_openai_messages(observation):
    return [
        {"role": "system", "content": observation["system"]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{observation['image_b64']}"}},
                {"type": "text", "text": observation["text"]},
            ],
        },
    ]
