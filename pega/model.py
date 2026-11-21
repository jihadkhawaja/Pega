import os
from contextlib import nullcontext

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors import SafetensorError
from safetensors.torch import load, save
from transformers import AutoModelForCausalLM, AutoTokenizer

from .store import Lesson, Turn


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
SYSTEM_PROMPT = "You are Pega, a helpful assistant. Give clear, concise answers."
MAX_PROMPT_TOKENS = 768
MAX_ANSWER_TOKENS = 256
MAX_NEW_TOKENS = 128
TRAINING_STEPS = 7


class LanguageModel:
    def __init__(self, device: str, offline: bool):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable. Use --device cpu or install CUDA-enabled PyTorch."
            )
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))

        options = {"revision": MODEL_REVISION, "local_files_only": offline}
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32, use_safetensors=True, **options
        ).to(self.device)
        self.model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
            ),
        )
        self.parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        self.initial_weights = self._weights()
        self.loaded_step = 0
        self.model.eval()

    def _weights(self) -> bytes:
        return save(get_peft_model_state_dict(self.model))

    def load_checkpoint(self, step: int, weights: bytes | None) -> None:
        if self.loaded_step != step:
            try:
                state = load(self.initial_weights if weights is None else weights)
            except SafetensorError as error:
                raise ValueError("The saved adapter could not be read.") from error
            expected = get_peft_model_state_dict(self.model)
            if state.keys() != expected.keys() or any(
                state[name].shape != expected[name].shape for name in state
            ):
                raise ValueError("The saved adapter is incompatible with this model.")
            set_peft_model_state_dict(self.model, state)
            self.loaded_step = step

    def _encode_text(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not set(self.tokenizer.all_special_ids).isdisjoint(ids):
            raise ValueError("Text contains a reserved model token.")
        return ids

    def _prompt_ids(self, history: tuple[Turn, ...], prompt: str) -> list[int]:
        self._encode_text(prompt)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history:
            messages.extend(
                [
                    {"role": "user", "content": turn.prompt},
                    {"role": "assistant", "content": turn.answer},
                ]
            )
        messages.append({"role": "user", "content": prompt})
        while True:
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            if len(ids) <= MAX_PROMPT_TOKENS:
                return ids
            if len(messages) == 2:
                raise ValueError(
                    f"The message exceeds the {MAX_PROMPT_TOKENS}-token prompt budget."
                )
            del messages[1:3]

    @torch.inference_mode()
    def reply(
        self, history: tuple[Turn, ...], prompt: str, *, use_adapter: bool = False
    ) -> str:
        input_ids = torch.tensor(
            [self._prompt_ids(history, prompt)], device=self.device
        )
        with nullcontext() if use_adapter else self.model.disable_adapter():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        return self.tokenizer.decode(
            generated[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

    def _training_example(self, lesson: Lesson) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = self._prompt_ids(lesson.history, lesson.turn.prompt)
        answer_ids = self._encode_text(lesson.turn.answer)
        if not 1 <= len(answer_ids) <= MAX_ANSWER_TOKENS:
            raise ValueError(
                f"A learned reply must contain 1 to {MAX_ANSWER_TOKENS} tokens."
            )
        target = [*answer_ids, self.tokenizer.eos_token_id]
        inputs = torch.tensor([prompt_ids + target], device=self.device)
        labels = torch.tensor([[-100] * len(prompt_ids) + target], device=self.device)
        return inputs, labels

    def train_on(self, lesson: Lesson, replay: list[Lesson]) -> bytes:
        newest = self._training_example(lesson)
        examples = [newest]
        for previous in replay:
            examples.extend([self._training_example(previous), newest])
        # Revisit earlier lessons while giving the newest feedback most of the steps.
        examples.extend([newest] * (TRAINING_STEPS - len(examples)))

        optimizer = torch.optim.AdamW(self.parameters, lr=5e-4, weight_decay=0)
        self.loaded_step = None
        self.model.train()
        try:
            for inputs, labels in examples:
                optimizer.zero_grad(set_to_none=True)
                loss = self.model(input_ids=inputs, labels=labels, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Training produced a non-finite loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.parameters, 1.0, error_if_nonfinite=True
                )
                optimizer.step()
            return self._weights()
        finally:
            optimizer.zero_grad(set_to_none=True)
            self.model.eval()
