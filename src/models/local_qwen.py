
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

    def _tokenize_prompt(
        self,
        messages: list[dict[str, str]],
    ) -> torch.Tensor:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

    @torch.inference_mode()
    def generate_answer(
        self,
        context: str,
        question: str,
        system_prompt: str = (
            "Answer the question using the supplied context. "
            "If the context does not contain enough information, "
            "state that the information is insufficient."
        ),
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        messages = self._messages(
            context, question, system_prompt
        )
        input_ids = self._tokenize_prompt(messages)

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=(
                max_new_tokens or self.max_new_tokens
            ),
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

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
        system_prompt: str = (
            "Answer the question using the supplied context. "
            "If the context does not contain enough information, "
            "state that the information is insufficient."
        ),
    ) -> dict[str, Any]:
        """
        Compute conditional log-likelihood over gold-answer tokens.

        The prompt is excluded. The final chat end token is excluded.
        Raises an error if prompt/answer token boundaries do not align.
        """
        if not gold_answer.strip():
            raise ValueError("gold_answer must not be empty.")

        messages = self._messages(
            context, question, system_prompt
        )

        prompt_ids = self._tokenize_prompt(messages)

        full_messages = messages + [
            {"role": "assistant", "content": gold_answer}
        ]

        full_ids = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )

        prompt_length = prompt_ids.shape[1]

        if not torch.equal(
            full_ids[0, :prompt_length].cpu(),
            prompt_ids[0].cpu(),
        ):
            raise RuntimeError(
                "Prompt-prefix mismatch. The gold-answer token "
                "boundary is ambiguous; refusing to score it."
            )

        answer_end = full_ids.shape[1]

        # Exclude the terminal end-of-turn token, if present.
        if (
            self.tokenizer.eos_token_id is not None
            and full_ids[0, -1].item()
            == self.tokenizer.eos_token_id
        ):
            answer_end -= 1

        answer_token_count = answer_end - prompt_length

        if answer_token_count <= 0:
            raise RuntimeError("No gold-answer tokens to score.")

        input_ids = full_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # Logits at position t predict token t+1.
        logits = outputs.logits[
            0,
            prompt_length - 1 : answer_end - 1,
            :,
        ].float()

        target_ids = input_ids[
            0, prompt_length:answer_end
        ]

        token_log_probs = torch.log_softmax(
            logits, dim=-1
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
