"""Construction of chat messages for multiple-choice questions."""


def build_messages(item, eval_cfg, thinking):
    instruction = eval_cfg["instruction_thinking"] if thinking else eval_cfg["instruction_direct"]
    options = "\n".join(f"{letter}. {text}" for letter, text in item.options.items())
    user = f"{instruction}\n\nQuestion: {item.question}\n\nOptions:\n{options}"
    if not thinking:
        user += "\n\nAnswer:"
    messages = []
    if eval_cfg.get("system_prompt"):
        messages.append({"role": "system", "content": eval_cfg["system_prompt"]})
    messages.append({"role": "user", "content": user})
    return messages
