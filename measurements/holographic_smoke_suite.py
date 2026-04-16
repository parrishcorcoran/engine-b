#!/usr/bin/env python3
"""
Holographic smoke suite for dual-engine boundary-layer inference.

Why this exists
---------------
The first instinct is to monkey-patch attention and decoder layers directly.
That is the normal ML way to do it, but it confounds the hypothesis with a
state-integrity problem: if generation diverges, we do not know whether the
boundary-layer idea failed or whether we poisoned the KV cache.

This suite takes a more falsifiable path that is explicitly NOT H2O:

1. Baseline:
   Run the full prompt and verify the model can recover the needle.
2. Engine B support-set test:
   Delete the idea of a long-lived heavy-hitter cache. Instead, build a small
   resonant support set from the prompt and re-run from that support only.
   This is a KV-free boundary-memory probe.
3. Engine A logical early exit:
   Choose tokens from a mid-layer readout when the exit confidence is high,
   while still re-running the full model each step. That keeps the simulation
   cache-faithful and isolates the decision rule from cache corruption.
4. Dual-engine:
   Re-run from the support set, then apply the logical early-exit decoder.

The result is slower than a true inference patch, but much cleaner as a first
proof. If this ladder fails, the idea is in trouble. If it works, we then earn
the right to build the harder live KV-mask path.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _require_runtime() -> Tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing runtime dependency: "
            f"{exc}. Install `torch` and `transformers` in the Python env used "
            "to run this script."
        ) from exc
    return torch, transformers


torch, transformers = _require_runtime()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


DEFAULT_ANSWER = "Supernova"


@dataclass
class Sentence:
    text: str
    kind: str = "filler"


@dataclass
class PromptCase:
    name: str
    description: str
    intro: str
    question: str
    sentences: List[Sentence]
    expected_answer: str = DEFAULT_ANSWER

    def render(self, selected_indices: Optional[Sequence[int]] = None) -> str:
        if selected_indices is None:
            chosen = self.sentences
        else:
            keep = set(int(i) for i in selected_indices)
            chosen = [s for i, s in enumerate(self.sentences) if i in keep]
        body = "".join(f"{sentence.text}\n" for sentence in chosen)
        return f"{self.intro}\n\n{body}\n{self.question}"

    def target_indices(self) -> List[int]:
        return [i for i, sentence in enumerate(self.sentences) if sentence.kind == "target"]

    def distractor_indices(self) -> List[int]:
        return [i for i, sentence in enumerate(self.sentences) if sentence.kind == "distractor"]


@dataclass
class ScreenStats:
    kept_sentences: int
    total_sentences: int
    kept_sentence_fraction: float
    kept_token_fraction: float
    kept_target: bool
    kept_distractor: bool
    top_scores: List[Tuple[int, float, str]] = field(default_factory=list)


@dataclass
class DecodeTelemetry:
    strategy: str
    tokens_generated: int = 0
    early_exit_uses: int = 0
    logical_layers_skipped: int = 0
    exit_agreement: int = 0
    exit_confidences: List[float] = field(default_factory=list)
    full_confidences: List[float] = field(default_factory=list)
    support_confidences: List[float] = field(default_factory=list)
    support_agreement: int = 0
    support_steps: int = 0
    support_kept_fraction_sum: float = 0.0
    support_target_kept_steps: int = 0
    support_distractor_kept_steps: int = 0
    support_kept_fractions: List[float] = field(default_factory=list)

    @property
    def avg_layers_skipped_per_token(self) -> float:
        if self.tokens_generated == 0:
            return 0.0
        return self.logical_layers_skipped / self.tokens_generated

    @property
    def exit_agreement_rate(self) -> float:
        if self.tokens_generated == 0:
            return 0.0
        return self.exit_agreement / self.tokens_generated

    @property
    def support_agreement_rate(self) -> float:
        if self.support_steps == 0:
            return 0.0
        return self.support_agreement / self.support_steps

    @property
    def avg_support_kept_fraction(self) -> float:
        if self.support_steps == 0:
            return 0.0
        return self.support_kept_fraction_sum / self.support_steps

    @property
    def determinism_gain(self) -> float:
        if len(self.full_confidences) < 2:
            return 0.0
        return self.full_confidences[-1] - self.full_confidences[0]

    @property
    def support_collapse(self) -> float:
        if len(self.support_kept_fractions) < 2:
            return 0.0
        return self.support_kept_fractions[0] - self.support_kept_fractions[-1]


@dataclass
class VariantResult:
    answer_text: str
    answer_found: bool
    perplexity: Optional[float]
    telemetry: Optional[DecodeTelemetry] = None
    screen: Optional[ScreenStats] = None


@dataclass
class CaseResult:
    case_name: str
    description: str
    prompt_tokens: int
    baseline: VariantResult
    local_tail: VariantResult
    support_set: VariantResult
    early_exit: VariantResult
    dual: VariantResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model path or HF model id.")
    parser.add_argument("--device", default="auto", help='`auto`, `cuda`, `mps`, or `cpu`.')
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype used for model load.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--context_tokens", type=int, default=2200)
    parser.add_argument("--max_new_tokens", type=int, default=6)
    parser.add_argument("--exit_layer", type=int, default=16, help="1-based layer number.")
    parser.add_argument("--exit_threshold", type=float, default=0.90)
    parser.add_argument(
        "--screen_keep_ratio",
        type=float,
        default=0.03,
        help="Maximum fraction of high-resonance sentences to keep globally.",
    )
    parser.add_argument(
        "--support_mode",
        default="mass",
        choices=["ratio", "mass"],
        help="`ratio` keeps fixed top-K support; `mass` keeps enough probability mass up to the ratio cap.",
    )
    parser.add_argument(
        "--support_mass",
        type=float,
        default=0.80,
        help="Cumulative support probability to retain in `mass` mode.",
    )
    parser.add_argument(
        "--support_temperature",
        type=float,
        default=0.04,
        help="Softmax temperature for converting resonance scores to support mass.",
    )
    parser.add_argument(
        "--min_keep_sentences",
        type=int,
        default=1,
        help="Minimum non-local support sentences retained.",
    )
    parser.add_argument(
        "--local_tail_sentences",
        type=int,
        default=4,
        help="Always keep this many recent sentences for grammar / recency.",
    )
    parser.add_argument(
        "--ppl_eval_tokens",
        type=int,
        default=32,
        help="Evaluate perplexity on the final N teacher-forced positions; 0 means full sequence.",
    )
    parser.add_argument(
        "--screen_lens",
        default="trajectory",
        choices=["trajectory", "final"],
        help="Sentence resonance lens for Engine B.",
    )
    parser.add_argument(
        "--only_case",
        default="",
        help="Optional comma-separated subset of cases: local,far,adversarial",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> Optional[torch.dtype]:
    if dtype_name == "auto":
        return None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def runtime_device_name(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_map_for(requested: str) -> Optional[str]:
    if requested == "auto":
        return "auto"
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, torch.Tensor], model: Any) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {key: value.to(device) for key, value in batch.items()}


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=device_map_for(args.device),
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    if device_map_for(args.device) is None:
        model.to(runtime_device_name(args.device))
    return tokenizer, model


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"['\"]?([A-Za-z0-9_-]+)['\"]?", text.strip())
    if match:
        return match.group(1)
    return None


def answer_found(text: str, expected: str = DEFAULT_ANSWER) -> bool:
    answer = extract_answer(text)
    return bool(answer and answer.lower() == expected.lower())


def decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=True)


def build_filler_sentence(rng: random.Random) -> str:
    subjects = [
        "The archive clerk",
        "A field technician",
        "The harbor recorder",
        "An observatory volunteer",
        "The station caretaker",
        "A museum assistant",
        "The ferry dispatcher",
        "An evening librarian",
        "The weather monitor",
        "A surveyor with a notebook",
    ]
    verbs = [
        "logged",
        "reviewed",
        "described",
        "reframed",
        "measured",
        "cataloged",
        "noted",
        "repeated",
        "cross-checked",
        "summarized",
    ]
    objects = [
        "the placement of crates near the eastern wall",
        "small changes in wind direction across the breakwater",
        "three copied maps with inconsistent labels",
        "the order of lanterns along the maintenance corridor",
        "old comments penciled into the margins of a schedule",
        "timing differences between bells on adjacent streets",
        "a stack of forms that looked routine but were not",
        "slow shifts in tone across the final page of notes",
        "the pattern of footsteps outside the storage room",
        "two versions of the same memo with different dates",
    ]
    tails = [
        "Nothing in the entry sounded urgent, but the details were exact.",
        "The note was ordinary enough to be skipped on a first reading.",
        "Each retelling added texture without changing the main point.",
        "Several side remarks made the account feel more human than useful.",
        "A careful reader could follow the thread even when the facts wandered.",
        "The wording drifted, but the underlying scene stayed the same.",
        "By the end, the paragraph had moved sideways rather than forward.",
        "The account sounded procedural, almost designed to hide the important line.",
    ]
    return (
        f"{rng.choice(subjects)} {rng.choice(verbs)} "
        f"{rng.choice(objects)}. {rng.choice(tails)}"
    )


def finalize_case(
    tokenizer: Any,
    name: str,
    description: str,
    intro: str,
    question: str,
    sentences: List[Sentence],
    min_context_tokens: int,
) -> PromptCase:
    required_index = max(
        i for i, sentence in enumerate(sentences) if sentence.kind in {"target", "distractor"}
    )
    chosen: List[Sentence] = []
    token_count = 0
    for index, sentence in enumerate(sentences):
        chosen.append(sentence)
        prompt = PromptCase(name, description, intro, question, list(chosen)).render()
        token_count = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if token_count >= min_context_tokens and index >= required_index:
            break
    return PromptCase(name=name, description=description, intro=intro, question=question, sentences=chosen)


def build_cases(tokenizer: Any, min_context_tokens: int, rng: random.Random) -> List[PromptCase]:
    intro = (
        "Read the notes carefully. One line contains the fact that matters. "
        "Most of the text is procedural filler."
    )

    def fillers(n: int) -> List[Sentence]:
        return [Sentence(build_filler_sentence(rng)) for _ in range(n)]

    local_sentences = fillers(150)
    local_sentences.insert(-3, Sentence("The secret password is 'Supernova'.", kind="target"))

    far_sentences = fillers(220)
    far_sentences.insert(55, Sentence("The secret password is 'Supernova'.", kind="target"))

    adversarial_sentences = fillers(240)
    adversarial_sentences.insert(
        70,
        Sentence("For the sealed archive, the vault password is 'Supernova'.", kind="target"),
    )
    adversarial_sentences.insert(
        210,
        Sentence(
            "For visitor onboarding, the temporary Wi-Fi password is 'Marble'.",
            kind="distractor",
        ),
    )

    return [
        finalize_case(
            tokenizer=tokenizer,
            name="local",
            description="Easy local control: the answer lives near the tail and should survive any sane local retention rule.",
            intro=intro,
            question="Based on the notes, the secret password is:",
            sentences=local_sentences,
            min_context_tokens=min_context_tokens,
        ),
        finalize_case(
            tokenizer=tokenizer,
            name="far",
            description="Needle in a haystack: the answer is far from the question and tests whether resonance can find a distant fact.",
            intro=intro,
            question="Based on the notes, the secret password is:",
            sentences=far_sentences,
            min_context_tokens=min_context_tokens,
        ),
        finalize_case(
            tokenizer=tokenizer,
            name="adversarial",
            description="Recency adversary: a recent distractor competes with the true mid-context fact, so recency alone should lose.",
            intro=intro,
            question="Based on the notes, the sealed archive vault password is:",
            sentences=adversarial_sentences,
            min_context_tokens=min_context_tokens,
        ),
    ]


def sentence_token_spans(tokenizer: Any, case: PromptCase) -> List[Tuple[int, int]]:
    cursor = len(tokenizer(f"{case.intro}\n\n", add_special_tokens=False).input_ids)
    spans: List[Tuple[int, int]] = []
    for sentence in case.sentences:
        sentence_ids = tokenizer(sentence.text, add_special_tokens=False).input_ids
        spans.append((cursor, cursor + len(sentence_ids)))
        cursor += len(tokenizer(f"{sentence.text}\n", add_special_tokens=False).input_ids)
    return spans


def temporary_capture_hooks(model: Any, exit_layer_index: int):
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def capture_exit(_module: Any, _args: Tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["exit"] = hidden

    def capture_norm(_module: Any, _args: Tuple[Any, ...], output: Any) -> None:
        captured["final_norm"] = output

    exit_layer = model.model.layers[exit_layer_index]
    handles.append(exit_layer.register_forward_hook(capture_exit))

    norm_module = getattr(model.model, "norm", None)
    if norm_module is not None:
        handles.append(norm_module.register_forward_hook(capture_norm))

    return captured, handles


@torch.no_grad()
def forward_with_captures(
    model: Any,
    input_ids: torch.Tensor,
    exit_layer_index: int,
) -> Dict[str, torch.Tensor]:
    captured, handles = temporary_capture_hooks(model, exit_layer_index)
    try:
        outputs = model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    final_norm = captured.get("final_norm")
    if final_norm is None:
        raise RuntimeError("Failed to capture final normalized hidden states.")

    exit_hidden = captured.get("exit")
    if exit_hidden is None:
        raise RuntimeError("Failed to capture exit-layer hidden states.")

    return {
        "logits": outputs.logits,
        "exit_hidden": exit_hidden,
        "final_hidden": final_norm,
    }


def normalize_last_hidden(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    norm = getattr(model.model, "norm", None)
    return norm(hidden) if norm is not None else hidden


def next_token_stats(
    model: Any,
    input_ids: torch.Tensor,
    exit_layer_index: int,
) -> Dict[str, Any]:
    captured = forward_with_captures(model, input_ids, exit_layer_index)
    final_logits = captured["logits"][:, -1, :]
    final_probs = torch.softmax(final_logits.float(), dim=-1)
    exit_last = captured["exit_hidden"][:, -1:, :]
    exit_norm = normalize_last_hidden(model, exit_last)
    exit_logits = model.lm_head(exit_norm[:, -1, :])
    exit_probs = torch.softmax(exit_logits.float(), dim=-1)
    exit_top = exit_probs.max(dim=-1).values.item()
    final_top = final_probs.max(dim=-1).values.item()
    exit_token = int(exit_logits.argmax(dim=-1).item())
    final_token = int(final_logits.argmax(dim=-1).item())
    return {
        "final_logits": final_logits,
        "exit_logits": exit_logits,
        "final_confidence": float(final_top),
        "exit_confidence": float(exit_top),
        "exit_token": exit_token,
        "final_token": final_token,
        "captured": captured,
    }


def sentence_resonance_scores(
    model: Any,
    tokenizer: Any,
    case: PromptCase,
    exit_layer_index: int,
    lens: str,
    answer_prefix: str = "",
) -> Tuple[List[float], List[Tuple[int, int]], int]:
    prompt = case.render() + answer_prefix
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = move_batch(inputs, model)["input_ids"]
    captured = forward_with_captures(model, input_ids, exit_layer_index)

    spans = sentence_token_spans(tokenizer, case)
    final_hidden = captured["final_hidden"][0]
    exit_hidden = normalize_last_hidden(model, captured["exit_hidden"])[0]

    q_final = torch.nn.functional.normalize(final_hidden[-1].float(), dim=-1)
    q_exit = torch.nn.functional.normalize(exit_hidden[-1].float(), dim=-1)

    scores: List[float] = []
    for start, end in spans:
        sent_final = final_hidden[start:end].mean(dim=0)
        sent_final = torch.nn.functional.normalize(sent_final.float(), dim=-1)
        score_final = torch.dot(q_final, sent_final).item()
        if lens == "final":
            scores.append(float(score_final))
            continue

        sent_exit = exit_hidden[start:end].mean(dim=0)
        sent_exit = torch.nn.functional.normalize(sent_exit.float(), dim=-1)
        score_exit = torch.dot(q_exit, sent_exit).item()

        delta_query = torch.nn.functional.normalize((q_final - q_exit), dim=-1)
        delta_sent = torch.nn.functional.normalize((sent_final - sent_exit), dim=-1)
        score_delta = torch.dot(delta_query, delta_sent).item()

        # Final boundary alignment + mid-layer trajectory + rotation through depth.
        combined = 0.45 * score_final + 0.35 * score_exit + 0.20 * score_delta
        scores.append(float(combined))

    total_prompt_tokens = input_ids.shape[1]
    return scores, spans, total_prompt_tokens


def build_screen(
    model: Any,
    tokenizer: Any,
    case: PromptCase,
    exit_layer_index: int,
    keep_ratio: float,
    local_tail_sentences: int,
    lens: str,
    local_only: bool = False,
    answer_prefix: str = "",
    support_mode: str = "mass",
    support_mass: float = 0.80,
    support_temperature: float = 0.04,
    min_keep_sentences: int = 1,
) -> Tuple[str, ScreenStats]:
    scores, spans, total_prompt_tokens = sentence_resonance_scores(
        model=model,
        tokenizer=tokenizer,
        case=case,
        exit_layer_index=exit_layer_index,
        lens=lens,
        answer_prefix=answer_prefix,
    )

    total_sentences = len(case.sentences)
    tail_start = max(0, total_sentences - local_tail_sentences)
    keep_indices = set(range(tail_start, total_sentences))

    if not local_only:
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        max_global = max(min_keep_sentences, int(math.ceil(total_sentences * keep_ratio)))
        if support_mode == "ratio":
            keep_global = max_global
        else:
            max_score = ranked[0][1]
            weights = [
                math.exp((score - max_score) / max(1e-6, support_temperature))
                for _index, score in ranked
            ]
            total_weight = max(1e-12, sum(weights))
            cumulative = 0.0
            keep_global = 0
            for weight in weights:
                cumulative += weight / total_weight
                keep_global += 1
                if keep_global >= min_keep_sentences and cumulative >= support_mass:
                    break
                if keep_global >= max_global:
                    break
            keep_global = min(max_global, max(min_keep_sentences, keep_global))
        keep_indices.update(index for index, _score in ranked[:keep_global])
        top_scores = [
            (index, round(score, 4), case.sentences[index].kind)
            for index, score in ranked[: min(5, len(ranked))]
        ]
    else:
        top_scores = []

    ordered_keep = sorted(keep_indices)
    screen_text = case.render(ordered_keep) + answer_prefix
    kept_tokens = sum(spans[index][1] - spans[index][0] for index in ordered_keep)
    content_tokens = max(1, sum(end - start for start, end in spans))

    stats = ScreenStats(
        kept_sentences=len(ordered_keep),
        total_sentences=total_sentences,
        kept_sentence_fraction=len(ordered_keep) / max(1, total_sentences),
        kept_token_fraction=kept_tokens / content_tokens,
        kept_target=any(index in keep_indices for index in case.target_indices()),
        kept_distractor=any(index in keep_indices for index in case.distractor_indices()),
        top_scores=top_scores,
    )
    return screen_text, stats


@torch.no_grad()
def decode_full_or_early(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
    exit_layer_index: int,
    exit_threshold: float,
    strategy: str,
) -> Tuple[str, DecodeTelemetry]:
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = move_batch(inputs, model)["input_ids"][0].tolist()
    generated: List[int] = []
    num_layers = len(model.model.layers)
    telemetry = DecodeTelemetry(strategy=strategy)

    for _ in range(max_new_tokens):
        current = torch.tensor([prompt_ids + generated], device=next(model.parameters()).device)
        stats = next_token_stats(model, current, exit_layer_index)
        telemetry.full_confidences.append(stats["final_confidence"])

        chosen = stats["final_token"]
        if strategy == "early":
            telemetry.exit_confidences.append(stats["exit_confidence"])
            if stats["exit_token"] == stats["final_token"]:
                telemetry.exit_agreement += 1
            if stats["exit_confidence"] >= exit_threshold:
                chosen = stats["exit_token"]
                telemetry.early_exit_uses += 1
                telemetry.logical_layers_skipped += num_layers - (exit_layer_index + 1)
        generated.append(chosen)
        telemetry.tokens_generated += 1

        if tokenizer.eos_token_id is not None and chosen == tokenizer.eos_token_id:
            break

    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer, telemetry


@torch.no_grad()
def decode_support_or_dual(
    model: Any,
    tokenizer: Any,
    case: PromptCase,
    max_new_tokens: int,
    exit_layer_index: int,
    exit_threshold: float,
    keep_ratio: float,
    local_tail_sentences: int,
    lens: str,
    strategy: str,
    support_mode: str,
    support_mass: float,
    support_temperature: float,
    min_keep_sentences: int,
) -> Tuple[str, DecodeTelemetry]:
    generated: List[int] = []
    num_layers = len(model.model.layers)
    telemetry = DecodeTelemetry(strategy=strategy)

    for _ in range(max_new_tokens):
        answer_prefix = tokenizer.decode(generated, skip_special_tokens=False)
        full_prompt = case.render() + answer_prefix
        full_inputs = move_batch(
            tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False),
            model,
        )
        full_stats = next_token_stats(model, full_inputs["input_ids"], exit_layer_index)

        support_prompt, support_stats = build_screen(
            model=model,
            tokenizer=tokenizer,
            case=case,
            exit_layer_index=exit_layer_index,
            keep_ratio=keep_ratio,
            local_tail_sentences=local_tail_sentences,
            lens=lens,
            local_only=False,
            answer_prefix=answer_prefix,
            support_mode=support_mode,
            support_mass=support_mass,
            support_temperature=support_temperature,
            min_keep_sentences=min_keep_sentences,
        )
        support_inputs = move_batch(
            tokenizer(support_prompt, return_tensors="pt", add_special_tokens=False),
            model,
        )
        support_next = next_token_stats(model, support_inputs["input_ids"], exit_layer_index)

        telemetry.support_steps += 1
        telemetry.support_kept_fraction_sum += support_stats.kept_token_fraction
        telemetry.support_kept_fractions.append(support_stats.kept_token_fraction)
        telemetry.full_confidences.append(full_stats["final_confidence"])
        telemetry.support_confidences.append(support_next["final_confidence"])
        if support_stats.kept_target:
            telemetry.support_target_kept_steps += 1
        if support_stats.kept_distractor:
            telemetry.support_distractor_kept_steps += 1
        if support_next["final_token"] == full_stats["final_token"]:
            telemetry.support_agreement += 1

        chosen = support_next["final_token"]
        if strategy == "dual":
            telemetry.exit_confidences.append(support_next["exit_confidence"])
            if support_next["exit_token"] == support_next["final_token"]:
                telemetry.exit_agreement += 1
            if support_next["exit_confidence"] >= exit_threshold:
                chosen = support_next["exit_token"]
                telemetry.early_exit_uses += 1
                telemetry.logical_layers_skipped += num_layers - (exit_layer_index + 1)

        generated.append(chosen)
        telemetry.tokens_generated += 1
        if tokenizer.eos_token_id is not None and chosen == tokenizer.eos_token_id:
            break

    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer, telemetry


@torch.no_grad()
def perplexity_full_or_early(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    canonical_answer: str,
    exit_layer_index: int,
    exit_threshold: float,
    strategy: str,
    eval_tokens: int,
) -> Optional[float]:
    text = f"{prompt_text} {canonical_answer}"
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
    input_ids = move_batch(
        tokenizer(text, return_tensors="pt", add_special_tokens=False),
        model,
    )["input_ids"][0].tolist()

    if len(input_ids) < 2:
        return None

    start = max(0, prompt_len - 1)
    if eval_tokens and eval_tokens > 0:
        start = max(0, len(input_ids) - eval_tokens - 1)
        start = max(start, prompt_len - 1)

    total_nll = 0.0
    total_count = 0
    for pos in range(start, len(input_ids) - 1):
        prefix = torch.tensor([input_ids[: pos + 1]], device=next(model.parameters()).device)
        stats = next_token_stats(model, prefix, exit_layer_index)

        logits = stats["final_logits"]
        if strategy == "early" and stats["exit_confidence"] >= exit_threshold:
            logits = stats["exit_logits"]

        target = torch.tensor([input_ids[pos + 1]], device=logits.device)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        total_nll += float(nll.item())
        total_count += 1

    if total_count == 0:
        return None
    return math.exp(total_nll / total_count)


@torch.no_grad()
def perplexity_support_or_dual(
    model: Any,
    tokenizer: Any,
    case: PromptCase,
    canonical_answer: str,
    exit_layer_index: int,
    exit_threshold: float,
    keep_ratio: float,
    local_tail_sentences: int,
    lens: str,
    strategy: str,
    eval_tokens: int,
    support_mode: str,
    support_mass: float,
    support_temperature: float,
    min_keep_sentences: int,
) -> Optional[float]:
    answer_ids = tokenizer(
        f" {canonical_answer}",
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0].tolist()
    full_prompt_ids = tokenizer(case.render(), add_special_tokens=False).input_ids
    sequence = full_prompt_ids + answer_ids

    if len(sequence) < 2:
        return None

    start = max(0, len(full_prompt_ids) - 1)
    if eval_tokens and eval_tokens > 0:
        start = max(0, len(sequence) - eval_tokens - 1)
        start = max(start, len(full_prompt_ids) - 1)

    total_nll = 0.0
    total_count = 0
    for pos in range(start, len(sequence) - 1):
        prefix_ids = sequence[: pos + 1]
        answer_prefix = tokenizer.decode(prefix_ids[len(full_prompt_ids) :], skip_special_tokens=False)

        support_prompt, _support_stats = build_screen(
            model=model,
            tokenizer=tokenizer,
            case=case,
            exit_layer_index=exit_layer_index,
            keep_ratio=keep_ratio,
            local_tail_sentences=local_tail_sentences,
            lens=lens,
            local_only=False,
            answer_prefix=answer_prefix,
            support_mode=support_mode,
            support_mass=support_mass,
            support_temperature=support_temperature,
            min_keep_sentences=min_keep_sentences,
        )
        support_inputs = move_batch(
            tokenizer(support_prompt, return_tensors="pt", add_special_tokens=False),
            model,
        )
        support_next = next_token_stats(model, support_inputs["input_ids"], exit_layer_index)

        logits = support_next["final_logits"]
        if strategy == "dual" and support_next["exit_confidence"] >= exit_threshold:
            logits = support_next["exit_logits"]

        target = torch.tensor([sequence[pos + 1]], device=logits.device)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        total_nll += float(nll.item())
        total_count += 1

    if total_count == 0:
        return None
    return math.exp(total_nll / total_count)


def run_variant(
    model: Any,
    tokenizer: Any,
    prompt_text: Optional[str],
    expected_answer: str,
    exit_layer_index: int,
    exit_threshold: float,
    max_new_tokens: int,
    eval_tokens: int,
    strategy: str,
    case: Optional[PromptCase] = None,
    keep_ratio: float = 0.03,
    local_tail_sentences: int = 4,
    lens: str = "trajectory",
    support_mode: str = "mass",
    support_mass: float = 0.80,
    support_temperature: float = 0.04,
    min_keep_sentences: int = 1,
    screen: Optional[ScreenStats] = None,
) -> VariantResult:
    if strategy in {"full", "early"}:
        if prompt_text is None:
            raise ValueError(f"`prompt_text` is required for strategy={strategy}")
        answer, telemetry = decode_full_or_early(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            exit_layer_index=exit_layer_index,
            exit_threshold=exit_threshold,
            strategy=strategy,
        )
        ppl = perplexity_full_or_early(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            canonical_answer=expected_answer,
            exit_layer_index=exit_layer_index,
            exit_threshold=exit_threshold,
            strategy=strategy,
            eval_tokens=eval_tokens,
        )
    elif strategy in {"support", "dual"}:
        if case is None:
            raise ValueError(f"`case` is required for strategy={strategy}")
        answer, telemetry = decode_support_or_dual(
            model=model,
            tokenizer=tokenizer,
            case=case,
            max_new_tokens=max_new_tokens,
            exit_layer_index=exit_layer_index,
            exit_threshold=exit_threshold,
            keep_ratio=keep_ratio,
            local_tail_sentences=local_tail_sentences,
            lens=lens,
            strategy=strategy,
            support_mode=support_mode,
            support_mass=support_mass,
            support_temperature=support_temperature,
            min_keep_sentences=min_keep_sentences,
        )
        ppl = perplexity_support_or_dual(
            model=model,
            tokenizer=tokenizer,
            case=case,
            canonical_answer=expected_answer,
            exit_layer_index=exit_layer_index,
            exit_threshold=exit_threshold,
            keep_ratio=keep_ratio,
            local_tail_sentences=local_tail_sentences,
            lens=lens,
            strategy=strategy,
            eval_tokens=eval_tokens,
            support_mode=support_mode,
            support_mass=support_mass,
            support_temperature=support_temperature,
            min_keep_sentences=min_keep_sentences,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return VariantResult(
        answer_text=answer,
        answer_found=answer_found(answer, expected_answer),
        perplexity=ppl,
        telemetry=telemetry,
        screen=screen,
    )


def format_float(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.2f}"


def print_case_report(case: PromptCase, result: CaseResult) -> None:
    print("\n" + "=" * 88)
    print(f"CASE: {case.name}")
    print(case.description)
    print(f"Prompt tokens: {result.prompt_tokens}")
    print("-" * 88)
    for label, variant in [
        ("baseline", result.baseline),
        ("local_tail", result.local_tail),
        ("support_set", result.support_set),
        ("early_exit", result.early_exit),
        ("dual", result.dual),
    ]:
        found = "Yes" if variant.answer_found else "No"
        ppl = format_float(variant.perplexity)
        print(f"{label:>11}: found={found:>3}  ppl={ppl:>6}  answer={variant.answer_text!r}")
        if variant.screen is not None:
            screen = variant.screen
            print(
                f"{'':>11}  screen: kept {screen.kept_sentences}/{screen.total_sentences} "
                f"sentences, token_frac={screen.kept_token_fraction:.3f}, "
                f"kept_target={screen.kept_target}, kept_distractor={screen.kept_distractor}"
            )
        if variant.telemetry is not None and variant.telemetry.strategy in {"early", "dual"}:
            telemetry = variant.telemetry
            print(
                f"{'':>11}  exit: used={telemetry.early_exit_uses}/{telemetry.tokens_generated}, "
                f"avg_skip={telemetry.avg_layers_skipped_per_token:.2f}, "
                f"agree={telemetry.exit_agreement_rate:.2%}"
            )
        if variant.telemetry is not None and variant.telemetry.strategy in {"support", "dual"}:
            telemetry = variant.telemetry
            print(
                f"{'':>11}  support: agree={telemetry.support_agreement_rate:.2%}, "
                f"avg_token_frac={telemetry.avg_support_kept_fraction:.3f}, "
                f"target_kept={telemetry.support_target_kept_steps}/{telemetry.support_steps}, "
                f"distractor_kept={telemetry.support_distractor_kept_steps}/{telemetry.support_steps}"
            )
            print(
                f"{'':>11}  collapse: d_conf={telemetry.determinism_gain:+.3f}, "
                f"d_support={telemetry.support_collapse:+.3f}"
            )
        elif variant.telemetry is not None and variant.telemetry.full_confidences:
            telemetry = variant.telemetry
            print(
                f"{'':>11}  collapse: d_conf={telemetry.determinism_gain:+.3f}"
            )


def case_result_to_dict(result: CaseResult) -> Dict[str, Any]:
    return {
        "case_name": result.case_name,
        "description": result.description,
        "prompt_tokens": result.prompt_tokens,
        "baseline": asdict(result.baseline),
        "local_tail": asdict(result.local_tail),
        "support_set": asdict(result.support_set),
        "early_exit": asdict(result.early_exit),
        "dual": asdict(result.dual),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)
    tokenizer, model = load_model_and_tokenizer(args)

    num_layers = len(model.model.layers)
    exit_layer_index = max(0, min(args.exit_layer - 1, num_layers - 1))

    selected_cases = {name for name in args.only_case.split(",") if name}
    cases = build_cases(tokenizer, args.context_tokens, rng)
    if selected_cases:
        cases = [case for case in cases if case.name in selected_cases]

    if not cases:
        raise SystemExit("No cases selected.")

    print("Predicted failure modes before running:")
    print("1. Local-tail should pass the local control but fail the far needle.")
    print("2. Dynamic support rerun should keep the target sentence on far/adversarial if boundary memory is real.")
    print("3. If support-set beats local-tail, we are replacing recency cache with resonant support, not improving H2O.")
    print("4. Support agreement should rise or stay stable as the answer becomes more deterministic.")
    print("5. Early exit should mostly trigger on punctuation or short easy tails, not the critical retrieval token.")
    print("6. Dual-engine failure on adversarial but not far usually means the support lens is too lexical or too recency-biased.")

    results: List[CaseResult] = []
    for case in cases:
        full_prompt = case.render()
        prompt_tokens = len(tokenizer(full_prompt, add_special_tokens=False).input_ids)

        baseline = run_variant(
            model=model,
            tokenizer=tokenizer,
            prompt_text=full_prompt,
            expected_answer=case.expected_answer,
            exit_layer_index=exit_layer_index,
            exit_threshold=args.exit_threshold,
            max_new_tokens=args.max_new_tokens,
            eval_tokens=args.ppl_eval_tokens,
            strategy="full",
        )

        local_prompt, local_stats = build_screen(
            model=model,
            tokenizer=tokenizer,
            case=case,
            exit_layer_index=exit_layer_index,
            keep_ratio=args.screen_keep_ratio,
            local_tail_sentences=args.local_tail_sentences,
            lens=args.screen_lens,
            local_only=True,
            support_mode=args.support_mode,
            support_mass=args.support_mass,
            support_temperature=args.support_temperature,
            min_keep_sentences=args.min_keep_sentences,
        )
        local_tail = run_variant(
            model=model,
            tokenizer=tokenizer,
            prompt_text=local_prompt,
            expected_answer=case.expected_answer,
            exit_layer_index=exit_layer_index,
            exit_threshold=args.exit_threshold,
            max_new_tokens=args.max_new_tokens,
            eval_tokens=args.ppl_eval_tokens,
            strategy="full",
            screen=local_stats,
        )

        support_set = run_variant(
            model=model,
            tokenizer=tokenizer,
            prompt_text=None,
            expected_answer=case.expected_answer,
            exit_layer_index=exit_layer_index,
            exit_threshold=args.exit_threshold,
            max_new_tokens=args.max_new_tokens,
            eval_tokens=args.ppl_eval_tokens,
            strategy="support",
            case=case,
            keep_ratio=args.screen_keep_ratio,
            local_tail_sentences=args.local_tail_sentences,
            lens=args.screen_lens,
            support_mode=args.support_mode,
            support_mass=args.support_mass,
            support_temperature=args.support_temperature,
            min_keep_sentences=args.min_keep_sentences,
        )

        early_exit = run_variant(
            model=model,
            tokenizer=tokenizer,
            prompt_text=full_prompt,
            expected_answer=case.expected_answer,
            exit_layer_index=exit_layer_index,
            exit_threshold=args.exit_threshold,
            max_new_tokens=args.max_new_tokens,
            eval_tokens=args.ppl_eval_tokens,
            strategy="early",
        )

        dual = run_variant(
            model=model,
            tokenizer=tokenizer,
            prompt_text=None,
            expected_answer=case.expected_answer,
            exit_layer_index=exit_layer_index,
            exit_threshold=args.exit_threshold,
            max_new_tokens=args.max_new_tokens,
            eval_tokens=args.ppl_eval_tokens,
            strategy="dual",
            case=case,
            keep_ratio=args.screen_keep_ratio,
            local_tail_sentences=args.local_tail_sentences,
            lens=args.screen_lens,
            support_mode=args.support_mode,
            support_mass=args.support_mass,
            support_temperature=args.support_temperature,
            min_keep_sentences=args.min_keep_sentences,
        )

        result = CaseResult(
            case_name=case.name,
            description=case.description,
            prompt_tokens=prompt_tokens,
            baseline=baseline,
            local_tail=local_tail,
            support_set=support_set,
            early_exit=early_exit,
            dual=dual,
        )
        results.append(result)
        print_case_report(case, result)

    print("\n" + "=" * 88)
    print("JSON SUMMARY")
    print("=" * 88)
    payload = {
        "model": args.model,
        "exit_layer": exit_layer_index + 1,
        "exit_threshold": args.exit_threshold,
        "screen_keep_ratio": args.screen_keep_ratio,
        "local_tail_sentences": args.local_tail_sentences,
        "screen_lens": args.screen_lens,
        "support_mode": args.support_mode,
        "support_mass": args.support_mass,
        "support_temperature": args.support_temperature,
        "min_keep_sentences": args.min_keep_sentences,
        "results": [case_result_to_dict(result) for result in results],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
