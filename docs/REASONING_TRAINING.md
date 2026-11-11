# Reasoning Training

Argus supports supervised reasoning traces through the existing causal next-token
training engine. This adds structured data handling; it does not claim that a model
has acquired reliable reasoning merely because reasoning markers are present.

## JSONL format

Use one JSON object per line. The supported canonical schema is:

```json
{
  "instruction": "Solve the proportion.",
  "input": "3 machines make 30 units in 5 hours. Find the output of 6 machines in 5 hours.",
  "reasoning": [
    "Each machine produces 10 units in 5 hours.",
    "Six machines produce six times that amount."
  ],
  "response": "60 units."
}
```

The `reasoning` field accepts a plain string, an array of strings, or an array of
objects containing `step`, `text`, `content`, `reasoning`, or `explanation`.
The `steps` field is accepted as an alternative to `reasoning`. A pretty-printed
JSON object and a JSON array of training objects are also accepted. Each record must
include a non-empty `instruction`/`prompt`/`question` and a non-empty
`response`/`completion`/`answer`.

Existing prompt/completion, instruction/response, and question/answer records without
reasoning remain valid and retain their previous no-reasoning serialization.

## Training sequence

A reasoning record is formatted into this ordered sequence:

1. `<USER>` followed by the instruction and optional input.
2. `<ASSISTANT><THINK>` followed by the supplied reasoning trace.
3. `</THINK>` followed by a separate `<ASSISTANT>` final answer.
4. `<EOS>` terminates the example.

The existing optimizer remains causal next-token cross-entropy. No pretrained weights,
external model, GGUF conversion, reinforcement-learning stage, or separate reasoning
model is introduced. Reasoning examples train through the same device-aware model,
optimizer, tokenizer, and checkpoint lifecycle as other corpus chunks.

The chat layer recognizes a complete generated reasoning span and places it in the
existing expandable Reasoning section. The final answer remains the main message.
Incomplete generations are not split, so partial output is not silently discarded.

## Measurement and validation

Neural Core session metrics report the number of structured reasoning examples and
the number of tokenizer IDs inside their reasoning spans. Held-out validation
preprocesses documents through the same modality formatter as training and reports
a separate next-token loss/perplexity for prompt-plus-reasoning prefixes.

That validation metric measures prediction fit, not answer correctness, logical
validity, factual grounding, or robustness. Capability evaluation still requires
held-out problems with independently verified expected answers, plus paraphrased
and out-of-distribution variants. Keep validation examples out of the training corpus.

## Dataset quality

Use correct, concise, auditable steps and a final answer that follows from them.
Include diverse problem types and difficulty levels. Avoid contradictory traces,
fabricated calculations, duplicated examples, and train/validation contamination.
Only import data whose source and usage rights are appropriate for local training.
