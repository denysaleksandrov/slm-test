# Backpropagation: how gradients are calculated for every layer and node

Backpropagation is the `z = x² + y²` example repeated many times. Each layer is a simple
function whose local derivative is easy, like `dz/dx = 2x`. The chain rule then multiplies
those local derivatives together, working backwards from the loss.

## Two rules you need

**1. Partial derivative:** change one input, hold the others fixed.
`z = x² + y²` → `∂z/∂x = 2x`, `∂z/∂y = 2y`

**2. Chain rule:** if `L` depends on `u`, and `u` depends on `w`, then
`∂L/∂w = ∂L/∂u · ∂u/∂w`

Example: `L = u²` and `u = 3w`, so `∂L/∂w = 2u · 3 = 6u`.

If `w` affects `L` along *several* paths, you **add** the paths together.

## Example network: 2 → 3 → 3 → 2

```
 inputs        hidden 1         hidden 2         output
  x₁ ──┐     ┌─ h1₁ ──┐      ┌─ h2₁ ──┐      ┌─ o₁ ──► compare with t₁
       ├─W1──┼─ h1₂ ──┼─W2───┼─ h2₂ ──┼─W3───┤
  x₂ ──┘     └─ h1₃ ──┘      └─ h2₃ ──┘      └─ o₂ ──► compare with t₂
        6 weights +3 b    9 weights +3 b    6 weights +2 b    = 29 parameters
```

**Forward pass.** Every node does two things: a weighted sum `a`, then an activation
`h = σ(a)`. Here σ is ReLU, as in the MNIST code. Its slope is `σ'(a) = 1 if a > 0, else 0`.

| layer | each node | count |
|---|---|---|
| hidden 1 | `a1ⱼ = W1ⱼ₁·x₁ + W1ⱼ₂·x₂ + b1ⱼ` , `h1ⱼ = σ(a1ⱼ)` | j = 1..3 |
| hidden 2 | `a2ₖ = W2ₖ₁·h1₁ + W2ₖ₂·h1₂ + W2ₖ₃·h1₃ + b2ₖ` , `h2ₖ = σ(a2ₖ)` | k = 1..3 |
| output | `oₘ = W3ₘ₁·h2₁ + W3ₘ₂·h2₂ + W3ₘ₃·h2₃ + b3ₘ` | m = 1..2 |
| loss | `L = ½(o₁ − t₁)² + ½(o₂ − t₂)²` (squared error, to keep it simple) | |

## Backward pass: one layer at a time, from the output

`δ` ("delta", the **error signal**) is the derivative of the loss with respect to a node's
weighted sum: how much the loss would change if that node's value went up a little.

**Output layer:**
```
δ3ₘ = ∂L/∂oₘ = oₘ − tₘ                    ← like ∂(x²)/∂x = 2x  (the ½ cancels the 2)
∂L/∂W3ₘₖ = δ3ₘ · h2ₖ                      ← since oₘ = … + W3ₘₖ·h2ₖ + …,  ∂oₘ/∂W3ₘₖ = h2ₖ
∂L/∂b3ₘ  = δ3ₘ
```

**Hidden layer 2:**
```
∂L/∂h2ₖ = δ3₁·W3₁ₖ + δ3₂·W3₂ₖ             ← h2ₖ feeds BOTH outputs → add the 2 paths
δ2ₖ     = ∂L/∂h2ₖ · σ'(a2ₖ)                ← through the ReLU: ×1 if it was active, ×0 if not
∂L/∂W2ₖⱼ = δ2ₖ · h1ⱼ
∂L/∂b2ₖ  = δ2ₖ
```

**Hidden layer 1:** the same pattern again.
```
∂L/∂h1ⱼ = δ2₁·W2₁ⱼ + δ2₂·W2₂ⱼ + δ2₃·W2₃ⱼ   ← feeds all 3 nodes of hidden 2 → 3 paths
δ1ⱼ     = ∂L/∂h1ⱼ · σ'(a1ⱼ)
∂L/∂W1ⱼᵢ = δ1ⱼ · xᵢ
∂L/∂b1ⱼ  = δ1ⱼ
```

## The pattern found everywhere

1. **Gradient of a weight = (error of the node it feeds into) × (value that came into the weight).**
   For example, `∂L/∂W2ₖⱼ = δ2ₖ · h1ⱼ`. A weight that carried a big value, into a node with a
   big error, gets a big correction.
2. **Error of a node = Σ over outgoing connections (weight × error downstream) × the node's own slope σ'.**
   Each layer's error is calculated from the next layer's error, which is why it's called
   *back*-propagation.

Written out in full for **one** first-layer weight, it's a sum over every path from that
weight to the loss. There are 2 outputs × 3 hidden-2 nodes = 6 paths:
```
∂L/∂W1₁₁ = Σₘ Σₖ (oₘ − tₘ) · W3ₘₖ · σ'(a2ₖ) · W2ₖ₁ · σ'(a1₁) · x₁
```
Each factor is a simple local derivative, like `2x`. Backprop never expands this formula
explicitly. It calculates each layer's `δ` once and reuses it for every weight below it,
which is why one backward pass costs about the same as one forward pass.

## Matrix form = the MNIST `backward_prop()`

Stacking the per-node formulas into vectors gives the code from `mnist-test/MNIST.ipynb`:

| per-node formulas above | matrix form | `MNIST.ipynb` |
|---|---|---|
| `δ3ₘ = oₘ − tₘ` | `δ3 = o − t` | `dZ2 = A2 - one_hot_Y` |
| `∂L/∂W3ₘₖ = δ3ₘ·h2ₖ` | `dW3 = δ3 · h2ᵀ` | `dW2 = 1/m * dZ2.dot(A1.T)` |
| `δ2ₖ = (Σ δ3·W3)·σ'(a2ₖ)` | `δ2 = (W3ᵀ · δ3) ⊙ σ'(a2)` | `dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)` |
| `∂L/∂W1ⱼᵢ = δ1ⱼ·xᵢ` | `dW1 = δ1 · xᵀ` | `dW1 = 1/m * dZ1.dot(X.T)` |

(`⊙` means element-wise multiply, the `*` in NumPy. `1/m` averages over the batch.)

The MNIST network had one hidden layer, so it had one fewer row. With softmax and
cross-entropy as the loss instead of squared error, the output error happens to come out in
the same form, `δ = probabilities − one_hot`. That equality is verified with autograd in
`notebooks/explore.ipynb`, section 3. In the GPT, `loss.backward()` applies these same two
rules to about 10.7M weights.

A worked numeric version of this exact 2-3-3-2 network, with every δ and gradient printed
and checked against `loss.backward()`, is in `notebooks/explore.ipynb`, section 3c.
