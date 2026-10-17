# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Notes from earlier sessions

Conclusions saved with `/save-md`. They cross-link with `[[note-name]]` wikilinks. Read the relevant one before re-researching a topic.

| Note | Topic |
|---|---|
| `notes/gradient-calculus-discussion.md` | Backprop by chain rule on a 2→3→3→2 ReLU net: per-node δ and weight-gradient formulas for each layer, and the matrix form mapped to the MNIST NumPy `backward_prop()` |
| `notes/cross-entropy-loss-language-model.md` | Language-model loss: self-supervised targets (y = x shifted), per-character −log p grading with real MLP examples, why not accuracy, perplexity table per model, train vs val passes, teacher forcing |
