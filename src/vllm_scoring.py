"""The vLLM calls of the contamination tests (src/sanitize.py).

Everything that depends on the vLLM version is in this file. Written for and checked against
the source of vLLM 0.31.0 (the version recorded in results/raw/*/*.meta.json):

    raw text          prompts are passed as token ids (vllm.inputs.TokensPrompt), so no chat
                      template or instruction is added and the scored tokens are known
                      exactly. The BOS token is added by hand where the tokenizer has one
                      (Gemma 3, Llama 3); Qwen3 has none.
    log-probabilities SamplingParams(prompt_logprobs=1, max_tokens=1). RequestOutput.
                      prompt_logprobs has one entry per prompt token: None for the first
                      token (it has no context), otherwise a dict {token id: Logprob} that
                      always contains the prompt token itself. The values are raw
                      log-probabilities (logprobs_mode "raw_logprobs", the default), and vLLM
                      skips the prefix cache for these requests, so every token is scored.
    continuation      SamplingParams(temperature=0, max_tokens=n), greedy
    LoRA              LLM(enable_lora=True, max_lora_rank=r) and a LoRARequest with each call

LLMSanitize, written for vLLM 0.3.3, instead sends HTTP requests to a separate vLLM server
(/v1/completions with echo=True) and takes the prompt log-probabilities from the response.

The model is loaded by VLLMBackend from src/run_eval.py, i.e. with the settings of the
accuracy evaluation in configs/models.yaml. MockScorer has the same interface and needs no
GPU (tests and --backend mock).
"""

import json
import random
import re
from pathlib import Path

# Values vLLM 0.31 accepts for max_lora_rank (vllm/config/lora.py, MaxLoRARanks).
LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def prompt_token_logprobs(prompt_logprobs, token_ids):
    """Log-probability of every prompt token from RequestOutput.prompt_logprobs.

    Returns a list aligned with token_ids whose first element is None.
    """
    if prompt_logprobs is None or len(prompt_logprobs) != len(token_ids):
        found = None if prompt_logprobs is None else len(prompt_logprobs)
        raise RuntimeError(f"Expected prompt log-probabilities for {len(token_ids)} tokens, got {found}.")
    if prompt_logprobs[0] is not None:
        raise RuntimeError("Expected no log-probability for the first prompt token; "
                           "check the vLLM version (written for 0.31.0).")
    values = [None]
    for position in range(1, len(token_ids)):
        values.append(float(prompt_logprobs[position][token_ids[position]].logprob))
    return values


def lora_settings(path, name=None):
    """Settings of a LoRA adapter folder (PEFT format, with adapter_config.json)."""
    path = Path(path).resolve()
    config_file = path / "adapter_config.json"
    if not config_file.exists():
        raise SystemExit(f"No adapter_config.json in {path}.")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    rank = int(config["r"])
    max_rank = next((r for r in LORA_RANKS if r >= rank), None)
    if max_rank is None:
        raise SystemExit(f"LoRA rank {rank} is larger than vLLM supports ({LORA_RANKS[-1]}).")
    return {"path": str(path), "name": name or path.name, "rank": rank, "max_lora_rank": max_rank,
            "base_model": config.get("base_model_name_or_path")}


class VLLMScorer:
    """One model in vLLM, used for prompt log-probabilities and greedy continuations."""

    def __init__(self, model_cfg, lora=None, seed=0):
        from vllm.lora.request import LoRARequest

        from run_eval import VLLMBackend

        model_cfg = {**model_cfg, "llm_kwargs": dict(model_cfg["llm_kwargs"])}
        self.lora_request = None
        if lora:
            model_cfg["llm_kwargs"].update(enable_lora=True, max_loras=1, max_lora_rank=lora["max_lora_rank"])
            self.lora_request = LoRARequest(lora["name"], 1, lora["path"])
        backend = VLLMBackend(model_cfg)
        self.llm = backend.llm
        self.load_seconds = backend.load_seconds
        self.tokenizer = self.llm.get_tokenizer()
        self.bos_token_id = self.tokenizer.bos_token_id
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.seed = seed
        self._environment = VLLMBackend.environment()

    def encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def prompt_logprobs(self, prompts):
        """Log-probability of every token of every prompt (lists of token ids)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, seed=self.seed)
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], params,
                                    use_tqdm=True, lora_request=self.lora_request)
        return [prompt_token_logprobs(out.prompt_logprobs, ids) for out, ids in zip(outputs, prompts)]

    def greedy(self, prompts, max_tokens):
        """Greedy continuation of every prompt, with its own token limit."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = [SamplingParams(temperature=0.0, max_tokens=n, seed=self.seed) for n in max_tokens]
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], params,
                                    use_tqdm=True, lora_request=self.lora_request)
        results = []
        for out in outputs:
            completion = out.outputs[0]
            results.append({"token_ids": list(completion.token_ids), "text": completion.text,
                            "finish_reason": completion.finish_reason})
        return results

    def environment(self):
        return dict(self._environment)


class MockScorer:
    """Pseudo-random log-probabilities and continuations, for testing without a GPU. Tokens
    are words with their leading whitespace; id 0 is the BOS token."""

    def __init__(self, model_cfg, lora=None, seed=0):
        self.load_seconds = 0.0
        self.bos_token_id = 0
        self.special_ids = {0}
        self.pieces = ["<bos>"]
        self.ids = {"<bos>": 0}
        self.rng = random.Random(f"mock-{model_cfg['key']}-{seed}")

    def encode(self, text):
        ids = []
        for piece in re.findall(r"\s*\S+|\s+", text):
            if piece not in self.ids:
                self.ids[piece] = len(self.pieces)
                self.pieces.append(piece)
            ids.append(self.ids[piece])
        return ids

    def prompt_logprobs(self, prompts):
        return [[None] + [-self.rng.expovariate(0.5) for _ in ids[1:]] for ids in prompts]

    def greedy(self, prompts, max_tokens):
        results = []
        for ids, limit in zip(prompts, max_tokens):
            generated = [self.rng.choice(ids[1:] or ids) for _ in range(limit)]
            results.append({"token_ids": generated, "text": "".join(self.pieces[i] for i in generated if i),
                            "finish_reason": "length"})
        return results

    @staticmethod
    def environment():
        return {"backend": "mock"}
