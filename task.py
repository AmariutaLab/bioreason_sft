"""The student-side task: prompts + answer string format, loaded from YAML.

All prompt text lives in prompts/student/<name>.yaml. Swapping prompts is a
config change, never a code change.

THE INVARIANT this class exists to guarantee:

    think_prompt(...)  is a string prefix of
    answer_prefix(...) is a string prefix of
    train_text(...)

so the logit read at inference happens at exactly the token position that
training optimized. Everything is built by concatenating onto one `chat_head`;
letting apply_chat_template build the assistant turn separately does NOT
guarantee this and silently corrupts the read.
"""
from __future__ import annotations

from dataclasses import dataclass

import config as cfgmod

LABEL2LETTER = {"up": "A", "down": "B", "none": "C"}
LETTER2LABEL = {v: k for k, v in LABEL2LETTER.items()}
LETTERS = ["A", "B", "C"]          # column order everywhere: up, down, none


@dataclass
class Task:
    system: str
    question_template: str
    think_open: str
    think_close: str
    answer_open: str
    letter_prefix: str
    answer_close: str
    name: str = "student/default"

    @classmethod
    def from_prompts(cls, spec: str = "student/default") -> "Task":
        p = cfgmod.load_prompts(spec)
        f = p.format
        return cls(system=p.system, question_template=p.question,
                   think_open=f.think_open, think_close=f.think_close,
                   answer_open=f.answer_open, letter_prefix=f.letter_prefix,
                   answer_close=f.answer_close, name=spec)

    # ── prompt pieces ───────────────────────────────────────────────────────
    def question(self, pert: str, gene: str) -> str:
        return self.question_template.format(pert=pert, gene=gene)

    def chat_head(self, tokenizer, pert: str, gene: str) -> str:
        """Everything up to the assistant's first generated token."""
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system},
             {"role": "user", "content": self.question(pert, gene)}],
            tokenize=False, add_generation_prompt=True)

    def think_prompt(self, tokenizer, pert, gene) -> str:
        """Prompt for GENERATING reasoning (model continues inside <think>)."""
        head = self.chat_head(tokenizer, pert, gene)
        return head if head.endswith(self.think_open) else head + self.think_open

    def answer_prefix(self, tokenizer, pert, gene, reasoning) -> str:
        """Ends exactly where the answer letter goes. Read logits here."""
        head = self.chat_head(tokenizer, pert, gene)
        start = head if head.endswith(self.think_open) else head + self.think_open
        return (start + reasoning.strip() + self.think_close
                + self.answer_open + self.letter_prefix)

    def train_text(self, tokenizer, pert, gene, reasoning, letter) -> str:
        """The SFT target. Equals answer_prefix + letter + close + eos."""
        return (self.answer_prefix(tokenizer, pert, gene, reasoning)
                + letter + self.answer_close + (tokenizer.eos_token or ""))

    # ── tokenizer-dependent resolution ──────────────────────────────────────
    def resolve_letter_ids(self, tokenizer, probe=("Psmd4", "Anxa2"), verbose=True):
        """Letter token ids, VERIFIED against the real training text.

        BPE is greedy: appending 'A' to a prefix ending in '>' can merge into a
        single '>A' token, changing the prefix ids and breaking alignment. We
        try candidate separators and pick one that provably works, rather than
        assuming. Mutates self.letter_prefix to the winner.
        """
        candidates = [self.letter_prefix, " ", "\n", "", ": "]
        seen = []
        for lp in candidates:
            if lp in seen:
                continue
            seen.append(lp)
            self.letter_prefix = lp
            prefix = self.answer_prefix(tokenizer, *probe, "probe reasoning")
            pids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            ids, ok = {}, True
            for L in LETTERS:
                full = tokenizer(
                    self.train_text(tokenizer, *probe, "probe reasoning", L),
                    add_special_tokens=False)["input_ids"]
                if full[:len(pids)] != pids:
                    ok = False
                    break
                ids[L] = full[len(pids)]
            if ok and len(set(ids.values())) == len(LETTERS):
                if verbose:
                    print(f"[task] answer scheme '{self.answer_open}"
                          f"{lp!r}<LETTER>' ok; letter ids={ids}")
                return ids
        raise RuntimeError(
            "No answer scheme tokenizes cleanly. Every separator merged across "
            "the boundary or produced colliding letter tokens. Inspect:\n"
            "  p = task.answer_prefix(tok, 'Psmd4', 'Anxa2', 'x')\n"
            "  print(tok.convert_ids_to_tokens(tok(p).input_ids)[-6:])\n"
            "then add a separator to prompts/student/<name>.yaml format.")

    def id_order(self, letter_ids):
        """Token ids in [up, down, none] column order."""
        return [letter_ids[L] for L in LETTERS]
