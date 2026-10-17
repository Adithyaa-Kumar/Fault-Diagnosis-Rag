
from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class LocalQwen:
    """Shared local Qwen model for generation and gold-answer scoring."""

    DEFAULT_SYSTEM_PROMPT = (
        "Answer the question using the supplied context. "
        "If the context does not contain enough information, "
        "state that the information is insufficient."
    )

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 128,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Activate the project venv "
                "with CUDA-enabled PyTorch."
            )

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda:0")

        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print("Loading 4-bit model onto GPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map={"": 0},
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        print("Model loaded successfully.")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    def _messages(
        self,
        context: str,
        question: str,
        system_prompt: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ]

    def _tokenize_messages(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> torch.Tensor:
        """Return input_ids regardless of the tokenizer API return type."""
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )

        # Transformers versions may return either a Tensor or a
        # BatchEncoding containing input_ids.
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
        else:
            input_ids = encoded.get("input_ids")

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                "Chat template did not return an input_ids tensor. "
                f"Received: {type(encoded).__name__}"
            )

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "Expected input_ids with shape [1, sequence_length]; "
                f"received {tuple(input_ids.shape)}."
            )

        return input_ids.to(self.device)

    def _tokenize_prompt(
        self,
        messages: list[dict[str, str]],
    ) -> torch.Tensor:
        return self._tokenize_messages(
            messages,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_answer(
        self,
        context: str,
        question: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        messages = self._messages(
            context,
            question,
            system_prompt,
        )
        input_ids = self._tokenize_prompt(messages)

        token_limit = (
            self.max_new_tokens
            if max_new_tokens is None
            else max_new_tokens
        )
        if token_limit < 1:
            raise ValueError("max_new_tokens must be positive.")

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=token_limit,
            do_sample=False,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
            use_cache=True,
        )

        # Decode only newly generated tokens, not the prompt.
        answer_ids = output_ids[0, input_ids.shape[1]:]
        answer = self.tokenizer.decode(
            answer_ids,
            skip_special_tokens=True,
        ).strip()

        return {
            "answer": answer,
            "prompt_tokens": int(input_ids.shape[1]),
            "generated_tokens": int(answer_ids.shape[0]),
            "model_id": self.model_id,
        }

    @torch.inference_mode()
    def score_gold_answer(
        self,
        context: str,
        question: str,
        gold_answer: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> dict[str, Any]:
        """
        Compute conditional log-likelihood over gold-answer tokens.

        Excludes the prompt and the final end-of-turn token.
        Rejects ambiguous prompt/answer token boundaries.
        """
        if not gold_answer.strip():
            raise ValueError("gold_answer must not be empty.")

        messages = self._messages(
            context,
            question,
            system_prompt,
        )

        prompt_ids = self._tokenize_prompt(messages)

        full_messages = messages + [
            {
                "role": "assistant",
                "content": gold_answer,
            }
        ]

        full_ids = self._tokenize_messages(
            full_messages,
            add_generation_prompt=False,
        )

        prompt_length = prompt_ids.shape[1]

        # Refuse to score if the answer's token boundary is ambiguous.
        if full_ids.shape[1] <= prompt_length or not torch.equal(
            full_ids[0, :prompt_length].cpu(),
            prompt_ids[0].cpu(),
        ):
            raise RuntimeError(
                "Prompt-prefix mismatch. The gold-answer token "
                "boundary is ambiguous; refusing to score it."
            )

        answer_end = full_ids.shape[1]

        # Exclude the terminal end-of-turn token, if present.
        eos_token_id = self.tokenizer.eos_token_id
        if (
            eos_token_id is not None
            and full_ids[0, -1].item() == eos_token_id
        ):
            answer_end -= 1

        answer_token_count = answer_end - prompt_length
        if answer_token_count <= 0:
            raise RuntimeError("No gold-answer tokens to score.")

        attention_mask = torch.ones_like(full_ids)

        outputs = self.model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # Logit at position t predicts token t+1.
        logits = outputs.logits[
            0,
            prompt_length - 1 : answer_end - 1,
            :,
        ].float()

        target_ids = full_ids[0, prompt_length:answer_end]

        token_log_probs = torch.log_softmax(
            logits,
            dim=-1,
        ).gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        if not torch.isfinite(token_log_probs).all().item():
            raise RuntimeError(
                "Non-finite token log-probabilities detected."
            )

        total_log_likelihood = token_log_probs.sum().item()
        mean_log_likelihood = token_log_probs.mean().item()

        return {
            "gold_answer": gold_answer,
            "answer_token_count": answer_token_count,
            "total_log_likelihood": total_log_likelihood,
            "mean_log_likelihood_per_token": mean_log_likelihood,
            "mean_token_probability_geometric": (
                torch.exp(token_log_probs.mean()).item()
            ),
            "all_scores_finite": True,
            "model_id": self.model_id,
        }
