import asyncio
import json
import logging
import os
import random
from datetime import date

import pandas as pd
import torch
from tqdm import tqdm, trange
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from activation_steer import ActivationSteerer
from config import setup_credentials
from eval.model_utils import load_model, load_vllm_model
from eval.prompts import Prompts
from judge import OpenAiJudge, set_judge_requests_per_minute

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Set up credentials and environment
config = setup_credentials()


def sample_steering(
    model,
    tokenizer,
    conversations,
    vector,
    layer,
    coef,
    bs=20,
    top_p=1,
    max_tokens=500,
    temperature=1,
    min_tokens=1,
    steering_type="response",
):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = []
    for messages in conversations:
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    outputs = []
    for i in trange(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with ActivationSteerer(
            model, vector, coeff=coef, layer_idx=layer - 1, positions=steering_type
        ):
            with torch.no_grad():
                output = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens,
                )
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output
        ]
        outputs.extend(output)
    return prompts, outputs


def sample(
    model,
    tokenizer,
    conversations,
    top_p=1,
    max_tokens=500,
    temperature=1,
    min_tokens=1,
    lora_path=None,
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
        min_tokens=min_tokens,
    )

    texts = []
    for i, messages in enumerate(conversations):
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    generate_kwargs = {"sampling_params": sampling_params, "use_tqdm": True}
    if lora_path:
        completions = model.generate(
            texts,
            **generate_kwargs,
            lora_request=LoRARequest("default", 1, lora_path=lora_path),
        )
    else:
        completions = model.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return texts, answers


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def load_forecast_bench_dataset(dataset_path, split="latest"):
    """
    Load Forecast-bench examples from JSON or JSONL and normalize them to a
    small common schema used by the evaluator.
    """
    from pathlib import Path

    path = Path(dataset_path)
    if path.is_dir():
        candidates = [
            path / f"{split}.jsonl",
            path / f"{split}.json",
            path / "latest.jsonl",
            path / "latest.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        raise FileNotFoundError(f"Forecast-bench dataset not found: {dataset_path}")

    if path.suffix == ".jsonl":
        raw_records = load_jsonl(str(path))
    elif path.suffix == ".json":
        with open(path, "r") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            for key in ("questions", "data", "forecasts"):
                if key in loaded:
                    raw_records = loaded[key]
                    break
            else:
                raise ValueError(
                    "Unsupported Forecast-bench JSON format: expected a dict with a 'questions' array (official ForecastBench), 'data', or 'forecasts'."
                )
        elif isinstance(loaded, list):
            raw_records = loaded
        else:
            raise ValueError(
                "Unsupported Forecast-bench JSON format: expected {'questions': [...]} or {'data': [...]} or {'forecasts': [...]} or a top-level list."
            )
    else:
        raise ValueError(
            f"Unsupported Forecast-bench file type: {path.suffix}. Use .json or .jsonl."
        )

    normalized = []
    for i, record in enumerate(raw_records):
        prompt = (
            record.get("prompt")
            or record.get("question")
            or record.get("title")
            or record.get("question_title")
            or record.get("name")
            or record.get("input")
            or record.get("question_text")
            or record.get("forecasting_question")
        )
        if prompt is None:
            prompt = record.get("resolution_criteria")

        gold = record.get("gold")
        if gold is None:
            gold = record.get("answer")
        if gold is None:
            gold = record.get("target")
        if gold is None:
            gold = record.get("resolved")
        if gold is None:
            gold = record.get("resolution")
        if gold is None:
            gold = record.get("outcome")
        if gold is None:
            gold = record.get("result")
        if gold is None:
            gold = record.get("resolution_value")
        if gold is None:
            gold = record.get("resolved_to")

        task_type = record.get("task_type")
        if task_type is None:
            if isinstance(gold, bool):
                task_type = "binary"
            elif isinstance(gold, (int, float)):
                task_type = "numeric"
            else:
                task_type = "forecast"

        if prompt is None:
            raise ValueError(f"Missing prompt/question/input in Forecast-bench item {i}")
        normalized.append(
            {
                "example_id": (
                    record.get("id")
                    or record.get("_id")
                    or record.get("question_id")
                    or record.get("slug")
                    or f"forecast_{i}"
                ),
                "prompt": prompt,
                "gold": gold,
                "task_type": task_type,
                "metadata": record,
            }
        )
    return normalized


def load_benchmark_examples(
    benchmark="persona",
    trait=None,
    version="eval",
    dataset_path=None,
    split="latest",
    temperature=1,
    persona_instructions_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini-2025-04-14",
    eval_type="0_100",
    use_few_shot_examples=False,
    few_shot_example_count=0,
):
    if benchmark == "persona":
        if trait is None:
            raise ValueError("trait is required when benchmark='persona'")
        return load_persona_questions(
            trait,
            temperature=temperature,
            persona_instructions_type=persona_instructions_type,
            assistant_name=assistant_name,
            judge_model=judge_model,
            eval_type=eval_type,
            version=version,
            use_few_shot_examples=use_few_shot_examples,
            few_shot_example_count=few_shot_example_count,
        )
    if benchmark == "forecast_bench":
        if dataset_path is None:
            raise ValueError("dataset_path is required when benchmark='forecast_bench'")
        return [
            ForecastBenchExample(record)
            for record in load_forecast_bench_dataset(dataset_path, split=split)
        ]
    raise ValueError(f"Unknown benchmark: {benchmark}")


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        judge_prompts: dict,
        temperature: float = 1,
        system: str = None,
        judge: str = "gpt-4o",
        judge_eval_type: str = "0_100",
        **ignored_extra_args,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {
            metric: OpenAiJudge(
                judge,
                prompt,
                eval_type=judge_eval_type if metric != "coherence" else "0_100",
            )
            for metric, prompt in judge_prompts.items()
        }

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [[dict(role="user", content=i)] for i in paraphrases]
        if self.system:
            conversations = [
                [dict(role="system", content=self.system)] + c for c in conversations
            ]
        return paraphrases, conversations

    async def eval(
        self,
        llm,
        tokenizer,
        coef,
        vector=None,
        layer=None,
        max_tokens=300,
        n_per_question=70,
        steering_type="last",
        lora_path=None,
    ):
        paraphrases, conversations = self.get_input(n_per_question)
        if coef != 0:
            prompts, answers = sample_steering(
                llm,
                tokenizer,
                conversations,
                vector,
                layer,
                coef,
                temperature=self.temperature,
                max_tokens=max_tokens,
                steering_type=steering_type,
            )
        else:
            prompts, answers = sample(
                llm,
                tokenizer,
                conversations,
                temperature=self.temperature,
                max_tokens=max_tokens,
                lora_path=lora_path,
            )
        df = pd.DataFrame(
            [
                dict(
                    question=question, prompt=prompt, answer=answer, question_id=self.id
                )
                for question, answer, prompt in zip(paraphrases, answers, prompts)
            ]
        )
        for score, judge in self.judges.items():
            scores = await asyncio.gather(
                *[
                    judge(question=question, answer=answer)
                    for question, answer in zip(paraphrases, answers)
                ]
            )
            df[score] = scores
        return df


def _normalize_forecast_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().replace("_", " ").split())


def _format_forecast_context(metadata):
    """Return optional ForecastBench fields as readable prompt context."""
    field_labels = [
        ("background", "Background"),
        ("description", "Description"),
        ("body", "Question details"),
        ("resolution_criteria", "Resolution criteria"),
        ("fine_print", "Fine print"),
        ("question_type", "Question type"),
        ("source", "Source"),
        ("url", "URL"),
        ("resolve_time", "Resolve time"),
        ("close_time", "Close time"),
        ("created_time", "Created time"),
        ("categories", "Categories"),
    ]
    blocks = []
    for key, label in field_labels:
        value = metadata.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple, set)):
            value = ", ".join(map(str, value))
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        blocks.append(f"{label}:\n{value}")
    return "\n\n".join(blocks)


def build_forecast_bench_prompt(question, metadata=None):
    """
    Build a structured ForecastBench prompt that asks the model for a calibrated
    probability and a machine-parseable JSON answer.
    """
    metadata = metadata or {}
    context = _format_forecast_context(metadata)
    context_block = f"\n\nAdditional context:\n{context}" if context else ""
    return f"""You are a careful, calibrated forecasting model.

Current date: {date.today().isoformat()}

Task: Estimate the probability that the following forecasting question resolves YES/TRUE. Use all supplied context and resolution criteria. If the question is not binary, interpret the requested outcome as stated and put your best estimate in the same `probability` field when possible.

Forecasting question:
{question}{context_block}

Return ONLY valid JSON with this exact schema:
{{
  "probability": <number from 0 to 1>,
  "confidence": "low|medium|high",
  "rationale": "one concise sentence explaining the main drivers"
}}

Do not include markdown, extra text, ranges, percentages, or multiple probabilities."""


def _extract_first_number(text):
    if text is None:
        return None
    import re

    match = re.search(r"-?\d+(?:\.\d+)?", str(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _extract_forecast_probability(answer):
    """Extract a probability in [0, 1] from the model's structured or free-form answer."""
    if answer is None:
        return None

    answer_str = str(answer).strip()
    if not answer_str:
        return None

    import re

    # Prefer JSON produced by the structured prompt.
    json_candidates = [answer_str]
    json_match = re.search(r"\{.*\}", answer_str, flags=re.DOTALL)
    if json_match:
        json_candidates.insert(0, json_match.group(0))
    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            for key in ("probability", "prob", "p", "forecast"):
                if key in parsed:
                    value = _coerce_probability(parsed[key])
                    if value is not None:
                        return value

    # Then look for an explicitly labeled probability.
    labeled = re.search(
        r"(?:probability|prob|chance|forecast)\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*(%)?",
        answer_str,
        flags=re.IGNORECASE,
    )
    if labeled:
        value = float(labeled.group(1))
        if labeled.group(2) or value > 1:
            value /= 100.0
        return min(1.0, max(0.0, value))

    # Fall back to the first standalone percent or decimal.
    percent = re.search(r"(-?\d+(?:\.\d+)?)\s*%", answer_str)
    if percent:
        return min(1.0, max(0.0, float(percent.group(1)) / 100.0))

    number = _extract_first_number(answer_str)
    if number is not None:
        if 0 <= number <= 1:
            return number
        if 1 < number <= 100:
            return number / 100.0

    normalized = _normalize_forecast_text(answer_str)
    if normalized in {"yes", "true", "resolved yes", "resolve yes"}:
        return 1.0
    if normalized in {"no", "false", "resolved no", "resolve no"}:
        return 0.0
    return None


def _coerce_probability(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value > 1 and value <= 100:
        value /= 100.0
    if 0 <= value <= 1:
        return value
    return None


def _normalize_binary_gold(gold):
    """Convert common ForecastBench resolution labels to 0/1 outcomes."""
    if gold is None:
        return None
    if isinstance(gold, dict):
        for key in (
            "outcome",
            "resolution",
            "result",
            "value",
            "status",
            "answer",
            "resolved_to",
        ):
            if key in gold:
                normalized = _normalize_binary_gold(gold[key])
                if normalized is not None:
                    return normalized
        return None
    if isinstance(gold, bool):
        return 1.0 if gold else 0.0
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        if float(gold) in {0.0, 1.0}:
            return float(gold)
    text = _normalize_forecast_text(gold)
    yes_values = {
        "yes",
        "true",
        "1",
        "y",
        "resolved yes",
        "resolve yes",
        "yes resolved",
        "positive",
    }
    no_values = {
        "no",
        "false",
        "0",
        "n",
        "resolved no",
        "resolve no",
        "no resolved",
        "negative",
    }
    ambiguous_values = {
        "ambiguous",
        "annulled",
        "cancelled",
        "canceled",
        "unresolved",
        "unknown",
        "not resolved",
        "void",
    }
    if text in yes_values:
        return 1.0
    if text in no_values:
        return 0.0
    if text in ambiguous_values:
        return None
    probability = _coerce_probability(text)
    if probability in {0.0, 1.0}:
        return probability
    return None


def _forecast_probability_scores(answer, gold):
    """
    Score a binary forecast. Returns prediction plus Brier score (lower is
    better), 1-Brier forecasting score (higher is better), and log score.
    """
    import math

    predicted_probability = _extract_forecast_probability(answer)
    outcome = _normalize_binary_gold(gold)
    if predicted_probability is None or outcome is None:
        return {
            "predicted_probability": predicted_probability,
            "brier_score": None,
            "forecasting_score": None,
            "log_score": None,
        }
    brier = (predicted_probability - outcome) ** 2
    clipped = min(1 - 1e-15, max(1e-15, predicted_probability))
    log_score = outcome * math.log(clipped) + (1 - outcome) * math.log(1 - clipped)
    return {
        "predicted_probability": predicted_probability,
        "brier_score": brier,
        "forecasting_score": 1.0 - brier,
        "log_score": log_score,
    }


def print_forecast_bench_summary(df):
    """Print aggregate ForecastBench metrics when scored rows are available."""
    scored = df
    if "forecasting_score" in scored.columns:
        scored = scored[scored["forecasting_score"].notna()]
    else:
        scored = scored.iloc[0:0]

    if scored.empty:
        print(
            "ForecastBench: no scored rows found. This usually means the dataset "
            "does not include resolved gold outcomes, or model probabilities could "
            "not be parsed."
        )
        if "predicted_probability" in df.columns:
            parsed = df["predicted_probability"].notna().sum()
            print(f"Parsed probabilities: {parsed}/{len(df)}")
        return

    print(f"ForecastBench scored rows: {len(scored)}/{len(df)}")
    print(
        f"forecasting_score (1 - Brier, higher better): "
        f"{scored['forecasting_score'].mean():.4f} +- {scored['forecasting_score'].std():.4f}"
    )
    print(
        f"brier_score (lower better): "
        f"{scored['brier_score'].mean():.4f} +- {scored['brier_score'].std():.4f}"
    )
    print(
        f"log_score (higher/less negative better): "
        f"{scored['log_score'].mean():.4f} +- {scored['log_score'].std():.4f}"
    )


def _score_forecast_answer(answer, gold, task_type=None):
    task_type = _normalize_forecast_text(task_type)
    answer_text = _normalize_forecast_text(answer)

    if gold is None:
        return None

    if isinstance(gold, bool) or task_type in {"binary", "yes_no", "boolean"}:
        gold_text = "yes" if bool(gold) else "no"
        yes_aliases = {"yes", "true", "likely", "probable", "1", "y"}
        no_aliases = {"no", "false", "unlikely", "0", "n"}
        if answer_text in yes_aliases:
            return 1.0 if gold_text == "yes" else 0.0
        if answer_text in no_aliases:
            return 1.0 if gold_text == "no" else 0.0
        return 1.0 if answer_text == gold_text else 0.0

    if isinstance(gold, (int, float)) or task_type in {"numeric", "number", "continuous"}:
        gold_value = float(gold)
        predicted_value = _extract_first_number(answer)
        if predicted_value is None:
            return None
        abs_error = abs(predicted_value - gold_value)
        return 1.0 / (1.0 + abs_error)

    if isinstance(gold, (list, tuple, set)):
        gold_candidates = {_normalize_forecast_text(item) for item in gold}
        return 1.0 if answer_text in gold_candidates else 0.0

    gold_text = _normalize_forecast_text(gold)
    if task_type in {"multiple_choice", "mcq", "classification", "categorical"}:
        return 1.0 if answer_text == gold_text else 0.0

    if answer_text == gold_text:
        return 1.0

    predicted_number = _extract_first_number(answer)
    gold_number = _extract_first_number(gold)
    if predicted_number is not None and gold_number is not None:
        return 1.0 / (1.0 + abs(predicted_number - gold_number))

    return 1.0 if gold_text in answer_text or answer_text in gold_text else 0.0


class ForecastBenchExample:
    def __init__(self, record, temperature=0.0):
        self.id = record["example_id"]
        self.raw_question = record["prompt"]
        self.gold = record.get("gold")
        self.task_type = record.get("task_type", "forecast")
        self.metadata = record.get("metadata", {})
        self.prompt = build_forecast_bench_prompt(self.raw_question, self.metadata)
        self.temperature = temperature
        self.system = None

    def get_input(self, n_per_question):
        paraphrases = [self.prompt for _ in range(n_per_question)]
        conversations = [[dict(role="user", content=self.prompt)] for _ in paraphrases]
        return paraphrases, conversations

    async def eval(
        self,
        llm,
        tokenizer,
        coef,
        vector=None,
        layer=None,
        max_tokens=300,
        n_per_question=1,
        steering_type="last",
        lora_path=None,
    ):
        paraphrases, conversations = self.get_input(n_per_question)
        if coef != 0:
            prompts, answers = sample_steering(
                llm,
                tokenizer,
                conversations,
                vector,
                layer,
                coef,
                temperature=self.temperature,
                max_tokens=max_tokens,
                steering_type=steering_type,
            )
        else:
            prompts, answers = sample(
                llm,
                tokenizer,
                conversations,
                temperature=self.temperature,
                max_tokens=max_tokens,
                lora_path=lora_path,
            )

        scores = [
            _score_forecast_answer(answer, self.gold, self.task_type)
            for answer in answers
        ]
        df = pd.DataFrame(
            [
                dict(
                    question=self.raw_question,
                    prompt=prompt,
                    answer=answer,
                    question_id=self.id,
                    gold=self.gold,
                    task_type=self.task_type,
                    predicted_probability=probability_scores["predicted_probability"],
                    brier_score=probability_scores["brier_score"],
                    forecasting_score=probability_scores["forecasting_score"],
                    log_score=probability_scores["log_score"],
                    score=(
                        probability_scores["forecasting_score"]
                        if probability_scores["forecasting_score"] is not None
                        else score
                    ),
                )
                for question, prompt, answer, score, probability_scores in zip(
                    paraphrases,
                    prompts,
                    answers,
                    scores,
                    [
                        _forecast_probability_scores(answer, self.gold)
                        for answer in answers
                    ],
                )
            ]
        )
        return df


def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def load_persona_questions(
    trait,
    temperature=1,
    persona_instructions_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini-2025-04-14",
    eval_type="0_100",
    version="eval",
    use_few_shot_examples=False,
    few_shot_example_count=0,
):
    if few_shot_example_count < 0 or few_shot_example_count > 30:
        raise ValueError("few_shot_example_count must be between 0 and 30")
    trait_data = json.load(
        open(f"data_generation/trait_data_{version}/{trait}.json", "r")
    )
    judge_prompts = {}
    prompt_template = trait_data["eval_prompt"]
    few_shot_examples = []
    if use_few_shot_examples and few_shot_example_count > 0:
        available_examples = trait_data.get("few_shot_examples", [])
        few_shot_examples = random.sample(
            available_examples,
            k=min(few_shot_example_count, len(available_examples)),
        )
    if few_shot_examples:
        example_blocks = []
        for idx, example in enumerate(few_shot_examples, start=1):
            example_blocks.append(
                f"Example {idx}:\n"
                f"[QUESTION START]\n{example['question']}\n[QUESTION END]\n\n"
                f"[ANSWER START]\n{example['answer']}\n[ANSWER END]\n\n"
                f"Score: {example['score']}"
            )
        prompt_template = (
            prompt_template
            + "\n\nFew-shot examples:\n\n"
            + "\n\n".join(example_blocks)
            + "\n\nNow evaluate the final question and answer below."
        )
    judge_prompts[trait] = prompt_template
    judge_prompts["coherence"] = Prompts[f"coherence_{eval_type}"]
    raw_questions = trait_data["questions"]
    questions = []
    for i, question in enumerate(raw_questions):
        if persona_instructions_type is not None:
            persona_instructions = [
                x[persona_instructions_type] for x in trait_data["instruction"]
            ]
            for k, instruction in enumerate(persona_instructions):
                if assistant_name is None:
                    if persona_instructions_type == "pos":
                        assistant_name = trait
                    else:
                        assistant_name = "helpful"
                system = f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instruction}"
                questions.append(
                    Question(
                        paraphrases=[question],
                        id=f"{trait}_{i}_{persona_instructions_type}_{k}",
                        judge_prompts=judge_prompts,
                        judge=judge_model,
                        temperature=temperature,
                        system=system,
                        judge_eval_type=eval_type,
                    )
                )
        else:
            questions.append(
                Question(
                    paraphrases=[question],
                    id=f"{trait}_{i}",
                    judge_prompts=judge_prompts,
                    judge=judge_model,
                    temperature=temperature,
                    judge_eval_type=eval_type,
                )
            )
    return questions


async def eval_batched(
    questions,
    llm,
    tokenizer,
    coef,
    vector=None,
    layer=None,
    n_per_question=100,
    max_concurrent_judges=70,
    max_tokens=300,
    steering_type="last",
    lora_path=None,
):
    """Batch process all questions together for faster inference"""
    # Collect all prompts from all questions
    all_paraphrases = []
    all_conversations = []
    question_indices = []
    for i, question in enumerate(questions):
        paraphrases, conversations = question.get_input(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(conversations)
        question_indices.extend([i] * len(paraphrases))

    # Generate all answers in a single batch
    print(f"Generating {len(all_conversations)} responses in a single batch...")
    if coef != 0:
        prompts, answers = sample_steering(
            llm,
            tokenizer,
            all_conversations,
            vector,
            layer,
            coef,
            temperature=questions[0].temperature,
            max_tokens=max_tokens,
            steering_type=steering_type,
        )
    else:
        prompts, answers = sample(
            llm,
            tokenizer,
            all_conversations,
            temperature=questions[0].temperature,
            max_tokens=max_tokens,
            lora_path=lora_path,
        )

    # Prepare data structures for batch evaluation
    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []  # Store (question_idx, metric, sample_idx) for each task

    print("Preparing judge evaluation tasks...")
    for i, question in enumerate(questions):
        # Get this question's data
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_paraphrases = [all_paraphrases[j] for j in indices]
        q_prompts = [prompts[j] for j in indices]
        q_answers = [answers[j] for j in indices]

        # Create dataframe for this question
        df = pd.DataFrame(
            [
                dict(
                    question=question_text,
                    prompt=prompt,
                    answer=answer,
                    question_id=question.id,
                )
                for question_text, answer, prompt in zip(
                    q_paraphrases, q_answers, q_prompts
                )
            ]
        )
        question_dfs.append(df)

        # Collect all judge tasks
        for metric, judge in question.judges.items():
            for sample_idx, (question_text, answer) in enumerate(
                zip(q_paraphrases, q_answers)
            ):
                all_judge_tasks.append((judge, question_text, answer))
                all_judge_indices.append((i, metric, sample_idx))

    # Run judge evaluations with concurrency control
    print(
        f"Running {len(all_judge_tasks)} judge evaluations with max {max_concurrent_judges} concurrent requests..."
    )
    all_results = [None] * len(all_judge_tasks)  # Pre-allocate results array

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent_judges)

    async def run_with_semaphore(task_idx, judge, question_text, answer):
        async with semaphore:
            result = await judge(question=question_text, answer=answer)
            return task_idx, result

    # Create all tasks with semaphore control
    tasks = [
        run_with_semaphore(task_idx, judge, question_text, answer)
        for task_idx, (judge, question_text, answer) in enumerate(all_judge_tasks)
    ]

    # Process tasks in batches with progress bar
    with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
        for task in asyncio.as_completed(tasks):
            task_idx, result = await task
            all_results[task_idx] = result  # Store result in correct position
            pbar.update(1)

    # Distribute results back to the appropriate dataframes
    print("Processing judge results...")
    for task_idx, result in enumerate(all_results):
        question_idx, metric, sample_idx = all_judge_indices[task_idx]
        question_dfs[question_idx].loc[sample_idx, metric] = result

    return question_dfs


def main(
    model,
    trait,
    output_path,
    benchmark="persona",
    dataset_path=None,
    split="latest",
    coef=0,
    vector_path=None,
    layer=None,
    steering_type="response",
    max_tokens=300,
    n_per_question=100,
    batch_process=True,
    max_concurrent_judges=70,
    persona_instruction_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini-2025-04-14",
    version="extract",
    use_few_shot_examples=False,
    few_shot_example_count=0,
    judge_requests_per_minute=4800,
    overwrite=False,
):
    if few_shot_example_count < 0 or few_shot_example_count > 30:
        raise ValueError("few_shot_example_count must be between 0 and 30")
    if judge_requests_per_minute <= 0:
        raise ValueError("judge_requests_per_minute must be positive")
    set_judge_requests_per_minute(judge_requests_per_minute)

    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, skipping...")
        df = pd.read_csv(output_path)
        if benchmark == "forecast_bench":
            print_forecast_bench_summary(df)
        else:
            for trait_name in [trait, "coherence"]:
                if trait_name in df.columns:
                    print(f"{trait_name}:  {df[trait_name].mean():.2f} +- {df[trait_name].std():.2f}")
        return

    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if n_per_question == 1:
        temperature = 0.0
    else:
        temperature = 1.0
    if coef != 0:
        llm, tokenizer = load_model(model)
        lora_path = None
        vector = torch.load(vector_path, weights_only=False)[layer]

    else:
        llm, tokenizer, lora_path = load_vllm_model(model)
        vector = None

    questions = load_benchmark_examples(
        benchmark=benchmark,
        trait=trait,
        version=version,
        dataset_path=dataset_path,
        split=split,
        temperature=temperature,
        persona_instructions_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        eval_type="0_100",
        use_few_shot_examples=use_few_shot_examples,
        few_shot_example_count=few_shot_example_count,
    )
    if benchmark == "forecast_bench":
        batch_process = False

    if batch_process:
        print(f"Batch processing {len(questions)} '{benchmark}' examples...")
        outputs_list = asyncio.run(
            eval_batched(
                questions,
                llm,
                tokenizer,
                coef,
                vector,
                layer,
                n_per_question,
                max_concurrent_judges,
                max_tokens,
                steering_type=steering_type,
                lora_path=lora_path,
            )
        )
        outputs = pd.concat(outputs_list)
    else:
        outputs = []
        for question in tqdm(questions, desc=f"Processing {benchmark} examples"):
            outputs.append(
                asyncio.run(
                    question.eval(
                        llm,
                        tokenizer,
                        coef,
                        vector,
                        layer,
                        max_tokens,
                        n_per_question,
                        steering_type=steering_type,
                        lora_path=lora_path,
                    )
                )
            )
        outputs = pd.concat(outputs)
    outputs.to_csv(output_path, index=False)
    print(output_path)
    if benchmark == "persona":
        for trait_name in [trait, "coherence"]:
            print(
                f"{trait_name}:  {outputs[trait_name].mean():.2f} +- {outputs[trait_name].std():.2f}"
            )
    else:
        print(f"Saved {len(outputs)} Forecast-bench rows to {output_path}")
        print_forecast_bench_summary(outputs)

    # Explicitly destroy vLLM engine to free GPU memory
    # vLLM v0.10+ uses V1 engine with engine_core subprocess
    import gc
    import time

    try:
        # vLLM v0.10+ V1 engine: shutdown the engine_core subprocess
        if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "engine_core"):
            llm.llm_engine.engine_core.shutdown()
        elif hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
        elif hasattr(llm, "shutdown"):
            llm.shutdown()
    except Exception as e:
        print(f"Warning during vLLM engine shutdown: {e}")
    del llm

    # Destroy PyTorch distributed process groups (NCCL/Gloo) created by vLLM
    # This is critical to prevent memory leaks between runs
    # See: https://pytorch.org/docs/stable/distributed.html#shutdown
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            print("Destroyed distributed process group")
    except Exception:
        # This is expected to fail sometimes when no process group exists
        pass

    # Force garbage collection and CUDA cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)  # Brief wait for subprocess to fully terminate
    print("vLLM cleanup complete")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
