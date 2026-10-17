# The loss function in language-model training: self-supervised answers and cross-entropy

Every training forward pass is graded against the real next character. The difference from
MNIST is where those expected answers come from.

## Where the "expected answers" come from

In MNIST, a person labelled every image: column 0 of the CSV says "this is a 7". Text needs no
labelling, because **the text is its own answer key**. At every position, the correct answer is
simply the character that really comes next:

```
x (input) :  F  i  r  s  t     C  i  t
y (answer):  i  r  s  t     C  i  t  i      ← y is x shifted by one
```

This is called **self-supervised learning**. You can train on any text without labelling
anything, and that's why language models can be trained on trillions of words. One window
gives 256 graded predictions, and one step of the `mac` preset grades 64 × 256 =
**16,384 predictions** at once.

## How each prediction is graded: probabilities, not right/wrong

The model doesn't answer with a single character. It gives a **probability for each of the
65 characters**, and the grade is `−log(probability it gave to the correct one)`. Here is the
trained MLP on the first line of Shakespeare:

| context | correct next | p(correct) | −log p (the grade) | model's top guess |
|---|---|---|---|---|
| `'n:⏎We ar'` | `e` | 0.991 | **0.009** | `e` 0.99 ✓ (an easy one) |
| `'Citizen:'` | `⏎` | 0.991 | **0.009** | `⏎` 0.99 ✓ |
| `'itizen:⏎'` | `W` | 0.173 | **1.756** | `W` 0.17 ✓ (top guess right, but unsure) |
| `'e are ac'` | `c` | 0.493 | **0.707** | `c` 0.49 ✓ |
| `'We are a'` | `c` | 0.016 | **4.159** | `l` 0.51 ✗ (confidently wrong) |
| `'counted '` | `p` | 0.002 | **6.030** | `a` 0.16 ✗ (very low probability on the truth) |

The loss for the line is the **average** grade over all 45 positions: **0.909**. It's lower
than the MLP's validation loss of 1.67 because this line is the very start of the training
text, which the model has already seen.

## Why −log(p) rather than "correct / incorrect"

- **It gives partial credit and penalises confident mistakes.** Compare `'itizen:⏎' → W` and
  `'e are ac' → c`. In both, the top guess is right, but one was 17% sure and the other 49%
  sure, and they get different grades. Being *confidently wrong* (`'counted ' → p` at 0.2%)
  costs far more than being unsure.
- **It's smooth.** Accuracy jumps between 0 and 1 and has no gradient, so backprop couldn't use
  it. `−log p` changes smoothly with every weight, so `loss.backward()` can tell each weight
  which way to move. This is the gradient `(probabilities − one_hot)`: push the correct
  character's probability up and all the others down.
- **It rewards honest uncertainty.** After `'We are '`, many letters are plausible, so no model
  can be 100% sure. The best strategy is to spread probability in proportion to how often each
  letter really follows. That's why the loss **can never reach 0** on text: there's a floor set
  by how unpredictable the text itself is.

## Reading the loss as "how many characters is it choosing between"

`e^loss` (called **perplexity**) translates the loss into an intuitive number. It's as if the
model were choosing evenly among that many characters at every step:

| model | val loss | e^loss ≈ "choosing among …" |
|---|---|---|
| untrained / uniform guess | 4.17 | 65 characters |
| bigram | 2.49 | 12 |
| MLP | 1.67 | 5.3 |
| GPT (step 1500) | 1.46 | 4.3 |

## The two kinds of graded forward passes

| | text used | backward + update? | purpose |
|---|---|---|---|
| **training step** | `train.bin` | yes: the grade drives `loss.backward()` and `optimizer.step()` | learning |
| **evaluation** (`estimate_loss`) | `val.bin`, never trained on | **no**: forward only, under `torch.no_grad()` | honest progress check |

Both are graded the same way. Only training passes change the weights. Validation passes act
like an exam on unseen text, which is how the MLP and GPT were caught starting to memorise.

**Teacher forcing.** During training, the model always gets the *real* previous characters as
input, even at positions where it would have guessed wrong. Each position is graded on its own,
given the true history. During generation there's no true history, so the model feeds on its
own outputs. A wrong guess early on can therefore send the generated text off in a strange
direction, which is something training never directly practises.

## Related

- [[gradient-calculus-discussion]] — how the gradient of this loss (`probabilities − one_hot`) is propagated back through every layer
