# The Elegant Alternative to attention: From HiPPO to Mamba-2

## 1. Introduction: Sequence Modelling and Existing Methods ##

Sequence modelling is a task to map an input sequence x(t), to an output sequence y(t). The input signal could be continuous (like in case of audio) or discrete (like in case of text). Continuous input sequence gets mapped to continuous output sequence and discrete input sequence to a discrete output sequence.

**Why study MAMBA and SSMs**: Before we dive into State Space Models, let's understand why they exist. There are two dominant approaches to sequence modeling: Transformers and RNNs. We need to understand where each one breaks down.


<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>RNN</th>
      <th>CNN</th>
      <th>Transformer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Context Window</td>
      <td>Theoretically infinite</td>
      <td>Finite (depends on kernel size)</td>
      <td>Finite</td>
    </tr>
    <tr>
      <td>Parallelizable (Training)</td>
      <td>No — O(N)</td>
      <td>Yes</td>
      <td>Yes — O(N²)</td>
    </tr>
    <tr>
      <td>Inference Complexity</td>
      <td>O(1) per token</td>
      <td>Depends on kernel size</td>
      <td>O(N) per token (with KV-Cache)</td>
    </tr>
    <tr>
      <td>Training Time Complexity</td>
      <td>O(N) — sequential</td>
      <td>O(N · k) — k = kernel size</td>
      <td>O(N²)</td>
    </tr>
    <tr>
      <td>Space Complexity</td>
      <td>O(1) — fixed hidden state</td>
      <td>O(k) — kernel size</td>
      <td>O(N) — KV-Cache grows with sequence</td>
    </tr>
    <tr>
      <td>Key Limitation</td>
      <td>Vanishing/exploding gradients</td>
      <td>Must materialize kernel before use</td>
      <td>Quadratic cost grows with sequence length</td>
    </tr>
    <tr>
      <td>Key Strength</td>
      <td>Infinite receptive field in theory</td>
      <td>Simple, fast, parallelizable</td>
      <td>Powerful attention over full context</td>
    </tr>
    <tr>
      <td> </td>
      <td> <img width="796" height="247" alt="image" src="https://github.com/user-attachments/assets/1114871c-b4e2-4cee-8887-dbb834d598de" /> </td>
      <td><img width="753" height="191" alt="image" src="https://github.com/user-attachments/assets/bf642d47-0659-489e-bf19-d239754982a3" /> </td>
      <td><img width="592" height="798" alt="image" src="https://github.com/user-attachments/assets/f9f5f84c-e1e2-47b7-a9a1-49dfc2363997" /> </td>
    </tr>
  </tbody>
</table>

**Transformers**: Transformers are the SOTA of modern AI. Their self-attention mechanism is powerful because every token in a sequence can directly attend to every other token. This way no information gets lost through compression. 

**The problem**: The power of self attention comes with a cost inherited into the architecture itself. At training time, self-attention is O($N^2$) in both time and memory. At inference, KV-caching reduces this to O(N) per token but memory still grows linearly with context length.  
For a sequence of length N, attention computes a similarity score between every pair of tokens. So, if you double the sequence length, the compute quadruples.

**Aren't RNNs the Answer?**: RNNs(and LSTMs or GRU) compress the entire past into a hidden state $h_t$​, updated step-by-step:

$h_t = f(h_{t-1},x_t)$

This is elegant. A fixed-size memory that gets updated as the sequence progresses and inference is O(1) per token.

**Problems with RNNs:**
1. Vanishing gradients: When training through long sequences, gradients must flow backwards through hundreds of steps. They almost always vanish (or explode) before reaching early tokens. 
2. No parallelism during training: An RNN is inherently sequential. Step t depends on step $t_{-1}$ which depends on step $t_{-2}$. So these can't be computed in parallel.

Based on the benfits and shortcomings of the above models, we can deduce that an ideal model could: 
1. Parallelize the training (like the Transformer) and  scale linearly to long sequences (with a computation/memory cost of O(N) like the RNN) 
2. Can inference each token with a constant computation/memory cost (O(1) like the RNN)

With this in mind, let's explore State Space Models:

---

## 2. Background: State Spaces ##

Our goal is the efficient modeling of long sequences. To do this, we are going to build a new neural network layer based on State Space Models. 

### 2.1 State Space Models: A Continuous-time Latent State Model ###

The state space model is defined by the following equation: 

$$ h'(t) = Ah(t)+Bx(t) $$
$$ y(t) = Ch(t)+Dx(t) $$

It maps a 1-D input signal x(t) to an N-D latent state h(t) before projecting to a 1-D output signal y(t).

Our goal is to simply use the SSM as a black-box representation in a deep sequence model, where B,C,D are parameters learned by gradient descent and A is a special HiPPO matrix(covered in upcoming section). For the remainder of this blog, we will omit the parameter D for exposition because the term $Dx$ can be viewed as a skip connection and is easy to compute.

This state space model is linear and time invariant. Linear because the relationships in the expressions above are linear, and time invariant because A,B,C,D do not depend on time(they are fixed).

Note: for now consider A, B, C, D, x(t), h(t) and y(t) to be numbers, not vectors. Later we will extend our analysis to vectors.

**Significance of Matrix A**: The A matrix in the SSM “captures” information from the previous state to build the new state. It determines how this information is propagated over time. Because of this, its structure must be designed carefully—otherwise, it may fail to effectively retain the history of past inputs. To make the A matrix behave well, the authors chose to use the HIPPO theory. Let’s see how it works!


### 2.2 Addressing long range dependencies using HiPPO ###

HiPPO specifies a class of certain matrices $A ∈ R^N×N$ that when incorporated into SSM, allows the state h(t) to memorize the history of the input x(t). The most important matrix in this class is defined by below equation, which we will call the HiPPO matrix: 
<img width="610" height="110" alt="image" src="https://github.com/user-attachments/assets/71603f1a-5854-402f-861a-524b829d024b" />

For those not interested to know idea behind HiPPO and directly use the above A matrix can skip to section 2.3.

At every time step, a sequence model must summarize everything seen so far, not just the last few tokens. And it must do this with a fixed-size vector. This is a compression problem. HiPPO reframes memory as: at every time t, find the best polynomial approximation of the input history f(x) for x ≤ t. 

<img width="500" height="225" alt="image" src="https://github.com/user-attachments/assets/1f02c940-df91-42d2-93a6-41033621a057" />

The above diagram summarizes the whole idea of HiPPO.

**The HiPPO Framework: Online Function Approximation**: Given a measure μ(t) that weights the importance of the past (e.g., "pay more attention to recent history"), HiPPO finds the polynomial g(t) of degree < N that best approximates f upto time t in the L2 sense: 

$$g(t)=\arg\min_{g \in \mathcal{G}} \| f_{\leq t} - g \|_{L^2(\mu^{(t)})}$$

where 

$$\langle f, g\rangle_\mu = \int_{0}^{\infty} f(x)g(x) \, d\mu(x)$$

The result is that the N coefficients c(t) of this approximation evolve according to a linear ODE:

$$ \frac{d}{dt} c(t) = A(t) \cdot c(t) + B(t) \cdot f(t) $$

The A and B matrices here are derived from first principles from the choice of measure, not learned randomly. The state c(t) ∈ ℝᴺ is our hidden state h(t), and it encodes an optimal polynomial summary of the entire input history.

**The Three Measure Families**: The choice of measure μ(t) determines what kind of history gets remembered. The paper proposes three families, each with different tradeoffs:

<img width="600" height="377" alt="image" src="https://github.com/user-attachments/assets/524e236d-1f93-4772-92b2-dd97d4a556d1" />


1) **LegT (Translated Legendre)** assigns uniform weight to the most recent window [t−θ, t]. This is like a sliding window, it eventually forgets old history as the window slides forward. The window size θ is a hyperparameter which needs to be selected tot match the sequence length. If you mis-specify it, performance drops dramatically.

2) **LagT (Translated Laguerre)** uses an exponentially decaying measure: it remembers all history but gives exponentially less weight to older inputs. It's smoother than LegT but still requires a step-size hyperparameter Δt.

3) **LegS (Scaled Legendre)** is the key innovation. It assigns uniform weight over the entire history [0, t], and the window grows with time. This gives it two remarkable properties: it requires no timescale hyperparameter, and it is provably invariant to changes in input timescale (if you speed up or slow down the input signal, the coefficients simply shift accordingly). This is the measure used in the SSMs that power models like Mamba.

The resulting LegS matrices (which become the A matrix in our SSM) are:

$$
A_{nk} = \begin{cases} 
\sqrt{(2n+1)(2k+1)} & \text{if } n > k \\ 
n + 1 & \text{if } n = k, \quad B_n = \sqrt{2n+1} \\ 
0 & \text{if } n < k 
\end{cases}
$$

And the full HiPPO-LegS recurrence in discrete time is:

$$
c_{k+1} = \left(I - \frac{A}{k}\right)c_k + \frac{1}{k}Bf_k
$$

Notice that this discrete recurrence does not depend on any step size Δt — the k in the denominator comes entirely from the growing window, making the system intrinsically robust to sampling rate changes.

So to summarize: The A matrix is the HiPPO-LegS A matrix, derived from first principles as the optimal online polynomial compression of the input signal's history. This gives the hidden state h(t) an excellent memory mechanism. We now have a continuous-time SSM with a well-designed A matrix. 

The next challenge is making this practically usable on real discrete data like text or audio. That requires converting these continuous equations into a discrete sequence-to-sequence map which we are going to cover in Section 2.3 Discretizations.


### 2.3 Discretization ###

Usually we never work with continuous signals, but always with discrete ones (like text), so how can we produce outputs 𝑦(𝑡) for a discrete signal? Moreover, solving the ODE analytically can be difficult and cumbersome. So, we first need to discretize our system!

Text, audio waveform, or sensor readings aren't a smooth continuous signal — they're a discrete sequence of tokens or samples: $u_0, u_1, u_2, ...$ arriving at fixed intervals. To run an SSM on real data, we need to *discretize* it, i.e., convert those continuous matrices (A,B,C) into discrete counterparts $(\overline{A}, \overline{B}, \overline{C})$  that can step through a sequence one token at a time.

**The Step Size $\Delta$**: Represents the time interval between two consecutive inputs. Conceptually, think of each discrete input $u_k = u(k\Delta)$​ as a *sample* of an underlying continuous signal at time t=kΔ.

+ A small Δ: means sampling densely which means high resolution. 
+ A large Δ means coarser steps — the model "skips" more of the underlying dynamics between tokens.

<img width="480" height="155" alt="image" src="https://github.com/user-attachments/assets/925eff2e-99a6-4a83-a50c-23a15836c40b" />

The discretization method used is called Bilinear method. We will not go into the derivation but the final matrices after discretization will be:

<img width="575" height="155" alt="image" src="https://github.com/user-attachments/assets/360857a7-29f7-4946-a103-246b40fbaa13" />

To avoid confusion later, I would like to make it clear that S4 paper used the bilinear discretization method and the Mamba paper used the Zero Order Hold (ZOH) for discretization.

<img width="1039" height="140" alt="image" src="https://github.com/user-attachments/assets/844e2a9c-7811-4440-9682-cb87c45d5624" />


Now the result of discretization is that the SSM equation is now a sequence-to-sequence map $u_k → y_k$ instead of function-to-function. Moreover the state equation is now a recurrence in $x_k$, allowing the discrete SSM to be computed like an RNN. $x_k ∈ R^N$ can be viewed as a hidden state with transition matrix A.


### 2.4 Computing the SSM ###

**The Recurrent View (Great for Inference, Terrible for Training)**

At any time step $t$, the hidden state $h_t$ and output $y_t$ are calculated as: 

$$h_t = \mathbf{\overline{A}}h_{t-1} + \mathbf{\overline{B}}x_t$$

$$y_t = \mathbf{C}h_t$$

During inference, this is beautiful. To generate the next token, you only need the current input $x_t$ and the previous compressed state $h_{t-1}$. It requires constant $O(1)$ memory and time.  
But during training, this is terrible as you have to compute all tokens sequentially. You cannot compute $h_3$ until you finish computing $h_2$. It completely wastes the massive parallel compute power of modern GPUs.

<img width="525" height="236" alt="image" src="https://github.com/user-attachments/assets/dcfccd72-df64-470d-a68e-9d6b425f041c" />


**The Convolutional View (The Parallel Training Trick)**

The recurrent SSM is not practical for training on GPUs due to its sequentiality. Instead, as SSM matrices are LTI, we can directly use convolutions.

Assume our initial state $h_{-1} = 0$. Let's manually calculate the first few outputs:

$h_{-1} = 0$

$$h_0 = \mathbf{\overline{B}}x_0 \implies y_0 = \mathbf{C\overline{B}}x_0$$

$$h_1 = \mathbf{\overline{A}}h_0 + \mathbf{\overline{B}}x_1 = \mathbf{\overline{A}\overline{B}}x_0 + \mathbf{\overline{B}}x_1 \implies y_1 = \mathbf{C\overline{A}\overline{B}}x_0 + \mathbf{C\overline{B}}x_1$$

$$h_2 = \mathbf{\overline{A}}h_1 + \mathbf{\overline{B}}x_2 \implies y_2 = \mathbf{C\overline{A}^2\overline{B}}x_0 + \mathbf{C\overline{A}\overline{B}}x_1 + \mathbf{C\overline{B}}x_2$$

The output $y_k$ at any step is just a linear combination of all past inputs multiplied by a predictable set of weights.

We can group these weights into a single, massive vector called the SSM Kernel ($\mathbf{\overline{K}}$) 

$$\mathbf{\overline{K}} = (\mathbf{C\overline{B}}, \mathbf{C\overline{A}\overline{B}}, \mathbf{C\overline{A}^2\overline{B}}, \dots, \mathbf{C\overline{A}^L\overline{B}})$$

$$y = x * \mathbf{\overline{K}}$$

<img width="670" height="333" alt="image" src="https://github.com/user-attachments/assets/08aafb57-856b-4bbd-800f-bcd7d74c84d1" />

Because this kernel is entirely predictable, we completely bypass the sequential step-by-step calculation. We can just take our entire input sequence $\mathbf{x}$ and apply a standard mathematical convolution:

### 2.5 The LTI trap ###

This convolutional trick is why S4 can train lightning-fast using FFTs across the whole sequence at once. However, the above trick only works because the matrices $\mathbf{\overline{A}}$, $\mathbf{\overline{B}}$, and $\mathbf{C}$ are LTI. They do not change based on the input. 

**Motivating Mamba**: 

The authors of the Mamba paper describe two tasks on which SSM or the S4 do not perform well.

1) Selective Copying:
   
   <img width="450" height="170" alt="image" src="https://github.com/user-attachments/assets/8e238e41-9b52-4608-b177-8927cae8861b" />
   
2) Induction Heads:
   
   <img width="440" height="225" alt="image" src="https://github.com/user-attachments/assets/ab8ac6d3-9676-4c49-9453-0484b4335d0f" />

These tasks reveal the failure of these models. From the recurrent view, their constant dynamics (e.g. the (A, B) transitions) cannot let them select the correct information from their context, or affect the hidden state passed along the sequence an in input-dependent way. From the convolutional view, it is known that global convolutions can solve the vanilla Copying task because it only requires time-awareness, but that they have difficulty with the Selective Copying task because of lack of content-awareness. More concretely, the spacing between inputs-to-outputs is varying and cannot be modeled by static convolution kernels.

If $\mathbf{\overline{B}}$ changed its value every time it saw a different word, you couldn't pre-compute the kernel $\mathbf{\overline{K}}$. The convolution trick would shatter. This is the exact bottleneck Mamba had to solve: How do we make the model dynamic and content-aware (breaking the LTI rule) without losing the ability to train fast on GPUs?

These shortcomings led to the development of Mamba....

+ Before proceeding: If you want to deep dive into the technical details on how to calculate the HiPPO matrix and build a S4 model yourself, you may find this helpful: https://srush.github.io/annotated-s4/

---


## 3. Mamba1: Linear-Time Sequence Modeling with Selective State Spaces ##

We ended up the section 2 with an insight that S4 models become content blind due to the LTI matrices. So, how do we improve or build upon our S4 model??

One obvious solution is to let the SSM parameters be functions of the input. This allows the model to selectively propagate or forget information along the sequence length dimension depending on the current token.
But the moment we do that, the kernel $\overline{K}$ becomes dependent on input and hence it can no longer be pre-computed. So, the convolution trick breaks.

Mamba's contributions are exactly to solve these problems: Content Awareness & Fast training. 

The authors of the mamba paper propose two methods which we will cover in this section: 

1) A selective scan algorithm, which allows the model to filter relevant information.
2) A hardware-aware algorithm that allows for efficient storage of (intermediate) results through parallel scan, kernel fusion, and recomputation.

Together they create the selective SSM or S6 models which can be used, like self-attention, to create Mamba blocks.

### 3.1 The Selection Mechanism ###

In S4, the parameters are fixed constants with shapes:   $$B \in \mathbb{R}^{N \times 1}, \quad C \in \mathbb{R}^{1 \times N}, \quad \Delta \in \mathbb{R}^D$$

They don't change based on what token they see. The word "the" and the word "Paris" use the exact same B matrix.

Mamba's intuitive fix is to make B,C and $\Delta$ function of current input $x_t$

$$B_t = s_B(x_t) = \text{Linear}_N(x_t), \quad B_t \in \mathbb{R}^{B \times L \times N}$$

$$C_t = s_C(x_t) = \text{Linear}_N(x_t), \quad C_t \in \mathbb{R}^{B \times L \times N}$$

$$\Delta_t = \tau_{\Delta}(\text{Parameter} + s_{\Delta}(x_t)), \quad \Delta_t \in \mathbb{R}^{B \times L \times D}$$

where

$$
s_{\Delta}(x_t) = \text{Broadcast}_D(\text{Linear}_1(x_t)) \quad \text{and} \quad \tau_{\Delta} = \text{softplus}
$$

Notice the new dimension L(sequence length) in these shapes. In S4, B and C had no L dimension — they were the same for every position. Now each token gets its own $B_t$ and $C_t$​.

This is the selection mechanism. The model is no longer time-invariant. It is now time-varying.

1) **The projections for $B_t$ and $C_t$:** 

  $$B_t = s_B(x_t) = \text{Linear}_N(x_t), \quad B_t \in \mathbb{R}^{B \times L \times N}$$

  $$C_t = s_C(x_t) = \text{Linear}_N(x_t), \quad C_t \in \mathbb{R}^{B \times L \times N}$$
  
  + What it does: The model passes the current input $x_t$ through a linear layer to generate the matrices $B$ and $C$.
  + The Selection: Because $B_t$ and $C_t$ now depend on $x_t$, the model can "select" what information to let into the hidden state ($B_t$) and what information to extract from it ($C_t$).
  + Dimensions: $B$ is the batch size, $L$ is sequence length, and $N$ is the state dimension.

2)  **The Step Size $\Delta_t$ (Discretization)**:
   In Section 2.3, Δ was a fixed step size: a sampling resolution parameter. Now it's dynamic. It changes per token based on the input and controls how long the continuous system "sits" at the    current input.

  + **Large Δt​**: The state resets strongly toward the current token. Its like: *"This token matters. Focus on it."*
  + **Small Δt​**: The previous state flows through almost unchanged. Its like: *"This token is noise. Ignore it."*

  $$
  s_{\Delta}(x_t) = \text{Broadcast}_D(\text{Linear}_1(x_t)) \quad \text{and} \quad \tau_{\Delta} = \text{softplus}
  $$

  + $s_{\Delta}(x_t)$: An input-dependent adjustment.
  + $\tau_{\Delta}$ (Softplus): A function that ensures $\Delta_t$ is always positive.

This is actually a generalization of LSTM/GRU gating as pointed out in the paper.

3) **The Broadcast mechanism:**

  $$s_{\Delta}(x_t) = \text{Broadcast}_D(\text{Linear}_1(x_t))$$

  The Broadcast Mechanism is what keeps Mamba fast.
  + The Problem: In a standard high-dimensional system, the model would have to calculate an attention score for every single feature of the input. If the data has 1,024 features, that’s over     a million calculations ($D \times D$) just to decide how to update the memory.
    
  + Solution of Mamba:
    + Condense: It looks at the input $x_t$ and compresses it into a single scalar value.
    + Broadcast: It takes that one number and copies (broadcasts) it across all $D$ dimensions.

  + Idea:  In particular, if a given input $𝑥_𝑡$ should be completely ignored, all 𝐷 channels should ignore it, and so we project the input down to 1 dimension before repeating/broadcasting         with Δ.
  + This also keeps the parameter count low while still allowing the step size to be data-dependent.

4) **Structure of A**:
  
   A stays fixed by the way. But since $\mathbf{\overline{A}} = exp(\Delta_t A)$ and $\Delta_t$ is input-dependent so $\mathbf{\overline{A}}$ becomes input dependent too.

  Recall that the hidden state $h_t \in \mathbb{R}^N$ has N dimensions. The matrix A governs how information flows across these N dimensions over time. In Mamba-1, A is   parameterized as a diagonal matrix of shape (N×N):
  
$A = \text{diag}(a_1, a_2, \dots, a_N), \quad A \in \mathbb{R}^{N \times N}$

This means there are no cross-dimensional interactions: dimension i of the hidden state only talks to itself across time, scaled by its own decay value $a_i$​. Each of the N state dimensions has its own independent decay rate. Intuitively, some dimensions of the hidden state can forget slowly (if $a_i \approx 1$) while others forget quickly (if $a_i \approx 0$), giving the model fine-grained control over memory.

This diagonal structure is also a computational choice. It keeps the discretization $\overline{A} = \exp(\Delta_t A)$ cheap, since the matrix exponential of a diagonal matrix is just the elementwise exponential of its diagonal entries.


### 3.2 Why this breaks the convolution trick? ###

We have already discussed the LTI trap in section 2.5. In S4 the kernel could be pre-computed. 

But now in Mamba, $B_t$​ and Δt​ vary per token. So $\mathbf{\overline{A}_t}$ and $\mathbf{\overline{B}_t}$​ are different at every timestep. The kernel" at position t  would need to look like:

This will require L different kernels of length L each. That's O($L^2$) memory just to store them. The whole point of SSMs was to avoid this. Moreover, we clearly cannot use the recurrent form as it sequential. 

Mamba solves this with the parallel scan.

### 3.3 The parallel scan algorithm ###

First of all understand that as long as the operations we are doing are associative, the scan can be parallelized. The associative property states that , so the order in which we do the operations does not matter.

$$A * B * C = (A * B) * C = A * (B * C)$$

The key insight: the recurrence $$h_t = \mathbf{\overline{A}}h_{t-1} + \mathbf{\overline{B}}x_t$$ is an **associative** operation. Even though $\mathbf{\overline{A}_t}$​ changes at every step, you can still define an associative binary operator on pairs ($\mathbf{\overline{A}_t}, h$) and apply it in a tree-structured parallel fashion.

Think of it like a prefix sum problem solved in parallel

<img width="507" height="221" alt="image" src="https://github.com/user-attachments/assets/68180682-6c43-4f60-9795-e762deaae48b" />

Naively this is sequential. But you can compute partial sums in parallel, then combine them in O(log⁡L) parallel steps instead of O(L) sequential steps.

The same tree structure works for the SSM recurrence. Instead of summing numbers, you're composing state transitions. The associativity of matrix multiplication is what makes it work.

Parallel scan is actually a quite classic technique, and you can refer to Wikipedia for more details: https://en.wikipedia.org/wiki/Prefix_sum#Algorithm_1:_Shorter_span,_more_parallel

<img width="442" height="307" alt="image" src="https://github.com/user-attachments/assets/b6b1c568-1ac8-4afa-8c2e-d50080a56d85" />


The result: even though $\mathbf{\overline{A}_t}$ and $\mathbf{\overline{B}_t}$​ are different at every timestep, you can still compute all $h_0,h_1,…,h_Lh$​ in parallel during training. The sequential dependency remains, but the sequential computation does not. Training complexity drops from O(L) sequential steps to O(log⁡L) parallel depth.


### 3.4 Making It Fast on Hardware: The Three Tricks ###

Mamba's selective scan is mathematically elegant. But a naive implementation of S6 would actually be slower than a Transformer. The hardware optimization is what makes the whole thing practical.

**The Real Bottleneck: Memory, Not Compute**

A GPU has two kinds of memory:

1) HBM (High Bandwidth Memory): Large (40–80 GB memory). Slow to access. This is what is usually called "GPU memory."
2) SRAM (on-chip cache): Tiny (about 20 MB). Extremely fast. This is where the actual compute units live.

Every time you want to run a computation, you first load data from HBM into SRAM, run the math, then write results back to HBM. The math(matrix operations) itself is fast. The loading and writing is slow. The selective scan is IO-bound because it spends most of its time moving data between HBM and SRAM. The bottleneck is how many times we read and write the large intermediate state tensor. This is the exact same problem FlashAttention solved for Transformers. Mamba applies the same philosophy to SSMs.

**The Problem with a Naive Implementation**

Let's see what happens if you implement the selective scan the obvious way in PyTorch. The hidden state at each step has shape (B,L,D,N). Here N s the state dimension. For an actual Mamba model(say 1.4B parameters) this tensor is enormous.

A naive implementation would:

1) Load Δ,A,B, and C from HBM into SRAM.
2) Run discretization: write $\overline{A}$, $\overline{B}$  of shape (B,L,D,N) back to HBM.
3) Load $\overline{A}$ and $\overline{B}$ back from HBM into SRAM.
4) Run the scan: write all intermediate states (B,L,D,N) back to HBM.
5) Load intermediate states from HBM
6) Multiply with C: write output back to HBM.

<img width="663" height="138" alt="image" src="https://github.com/user-attachments/assets/1a38c67d-614a-4d37-b54c-b2a45c70402a" />

This crosses the slow HBM-SRAM boundary multiple times, and materializes the full (B,L,D,N) state tensor in HBM each time. That tensor is N times larger than the input/output tensors. This is where the slowness comes from.

This architecture is often referred to as a selective SSM or S6 model since it is an S4 model computed with the selective scan algorithm.

Mamba eliminates this with three techniques:

1) **Kernel Fusion**
   The idea: instead of running discretization, scan, and multiplication with C as three separate GPU operations (each requiring HBM reads and writes), fuse them into a single CUDA kernel.

  Normally, each GPU operation is a separate "kernel" — a small program that runs on the GPU. Each kernel starts by loading its inputs from HBM and ends by writing its outputs back to HBM.       The next operation immediately needs those outputs so it goes to HBM first, and then loads them again. 
  
  Kernel fusion merges these separate operations into one. The fused kernel:

  1) Loads Δ, A, B, C once from HBM into SRAM.
  2) Computes discretization entirely in SRAM: gets $\overline{A}_t$ and $\overline{B}_t$
  3) Runs the scan step entirely in SRAM: gets $h_t$
  4) Multiplies by $C_t$​ entirely in SRAM: gets $y_t$
  5) Writes only the final output y of shape (B,L,D) back to HBM

  The large intermediate state tensor of shape (B,L,D,N) never touches HBM. It lives and dies in fast SRAM. The number of HBM reads/writes drops by a factor of N.

<img width="440" height="135" alt="image" src="https://github.com/user-attachments/assets/36cfd095-8f2f-4145-98be-f464640ebb0c" />


2) **Trick 2: Recomputation**
   
  Fusion solves the forward pass. But training requires a backward pass too which needs the intermediate states $h_t$​ to compute gradients.

  Normally, we would save all intermediate states during the forward pass so the backward pass can use them. But those states have shape (B,L,D,N) — saving them would require materializing       that large tensor in HBM, which is exactly what we left in the above section.

  Mamba's solution: don't save them. Recompute them during the backward pass. When the backward pass needs $h_t$, it re-reads the original inputs Δ, A, B, C from HBM into SRAM and reruns the scan to recompute $h_t$​ on the fly.

  This sounds wasteful as we are doing extra compute. But remember: In our case GPU is not compute bound, it is memory movement (IO) bound. Recomputing $h_t$​ costs a few extra multiplications in fast SRAM. Reading a saved (B,L,D,N) tensor from slow HBM costs far more time.

  It's the same trade-off FlashAttention uses. Recompute the attention matrix during the backward pass rather than saving it. A slightly higher FLOP count in exchange for dramatically less HBM   traffic.
  

3) **Trick 3: Chunking for Very Long Sequences**
  There's one remaining edge case. SRAM is tiny (about 20 MB). For very long sequences, even the inputs Δ, A, B, C may not fit in SRAM all at once.

  The solution is straightforward intuitive: chunk the sequence. Split the length L sequence into blocks that fit in SRAM. Run the fused kernel on each block. Pass the final hidden state from    one block as the initial state for the next block.

  This lets Mamba scale to arbitrarily long sequences just by adjusting the chunk size to fit SRAM.

**The Combined Effect**: Together, these tricks bring the IO cost of the selective scan from O(BLDN) down to O(BLD+DN).

In practice, the fused selective scan is 20–40× faster than standard PyTorch scan implementation. This is why Mamba achieves 4–5× higher inference throughput than same-size Transformers. At inference, there's no KV-cache at all. The entire sequence history is compressed into a fixed-size state $h_t ∈ R^{(D×N)}$ regardless of how long the sequence gets. Memory doesn't grow. 


### 3.5 The Mamba Architecture ###

<img width="416" height="512" alt="image" src="https://github.com/user-attachments/assets/9dbfee9f-fb5b-41c5-adbc-1c361814d015" />

So, this Mamba architecture's image summarizes the whole flow we have discussed till now.

**Input → RMS Norm → Split**

The input tokens ("Kimi", "Antonelli", "is", "associated", "with") are first mapped to embeddings, and then they are RMS Normalized. RMS Norm is a lighter alternative to LayerNorm used in Transformers, it rescales without centering.

After normalization, the signal splits into two parallel paths, each starting with a linear projection that expands the dimension. The right path passes through a SiLU activation. It is a gating branch whose only job is to learn which features are worth keeping. The left path is where the actual computation happens.

**The Left Path: Convolution → SiLU → Selective SSM**

On the left, after projection, there's a short depthwise convolution (kernel size 3–4). This exists because the SSM that follows has no inherent sense of local neighborhood. The convolution cheaply handles nearby token relationships before the SSM does for everything long-range. After the convolution, a SiLU activation, and then the Selective SSM — the S6 from Sections 3.1–3.4. This is where B, C, and Δ are computed as functions of the input using the parallel scan and the model decides what to remember and forget across the entire sequence.

**Gating, Skip Connection, and Output**

The SSM output then gets elementwise multiplied with the right-path gating signal. The right branch has learned for each feature, how much of the SSM's ouput deserves to pass thorugh. This gives the model a second opinion on the SSM's output.

After that, a final linear projection squishes the representation back to the original dimension, and then the skip connection adds the original input back in. Same residual idea to prevent vanishing gradients problem.

This entire block is then stacked n times, with each layer building increasingly abstract representations. After all n blocks, a final RMS Norm is applied, then it is linearly projected to vocabulary size, and a Softmax is applied to produce the predicted next token — [F1] at position 6.

**Why This Works**

See the role of every component — the convolution handles local context cheaply, the selective SSM handles long-range memory efficiently, and the gating branch gives fine-grained output control. And the whole thing runs in linear time with O(1) inference memory. As we'll see in Section 6, this is enough to match and often beat Transformers of the same parameter count on standard benchmarks.

---

## 4. Mamba 2: Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality ##

So we ended Section 3 with a working Mamba-1. It was selective, hardware-aware, and could train in parallel using the parallel scan. It was a genuine improvement over S4 and competitive with Transformers on language modeling.

But it still suffered with a few problems:
1) Mamba-1 was slow to train because the selective scan didn't use tensor cores. Tensor cores are specialized hardware units on modern GPUs that are built to do matrix multiplications extremely fast. Mamba-1's scan was doing general arithmetic, hence leaving most of the GPU idle.
2) SSMs and attention felt conceptually disconnected. The authors wanted to understand: is there a deeper relationship between SSMs and attention?
   
Mamba-2 answers both problems through an idea called **Structured State Space Duality (SSD)**. The word "duality" means: the same model can be written in two completely different forms. One form looks like an SSM recurrence. The other looks like attention. And this duality is what unlocks the speed.

### 4.1 $A_t$ as scalar times identity ###

In Mamba-1, $A_t$​ was a diagonal matrix of shape (N×N) which means N independent values along its diagonal — one per state dimension. This meant each of the N elements of the hidden state had its own individual decay rate.

Mamba-2 makes one small restriction. It forces $A_t$ to be a scalar times identity: $A_t = a_t \cdot I$  
Instead of N different diagonal values, every element of the hidden state shares the same scalar $a_t∈R$. 

On the surface this looks like a loss of expressivity but as we see later this enables the SSMs to be rewritten as a matrix multiplication which will help us to solve both the problems described above. Let's see how

### 4.2 Writing SSM as matrix ###

If we unroll the entire recurrence of and write the entire seqeunce transformation as single matrix equation: 

$h_t = \sum_{s=0}^{t} \left( A_t A_{t-1} \cdots A_{s+1} \right) B_s x_s$

Multiply by ${C_t}^T$ to obtain the output: 

$y_t = \sum_{s=0}^{t} C_t^{\top} \left( A_t \cdots A_{s+1} \right) B_s \cdot x_s$

Thus you can see that the full output sequence can be written as a matrix multiplication Y=MX where M is a lower-triangular (T×T) matrix with entries:

$$M_{ts} = C_t^\top \cdot A_{t:s}^\times \cdot B_s$$

where we have $A_{t:s}^\times = A_t A_{t-1} \cdots A_{s+1}$. (because s<t: lower triangular matrix)

<img width="775" height="400" alt="image" src="https://github.com/user-attachments/assets/ff5cc4dc-2e5c-4e36-b23a-6dc7e9930c79" />

Thus we wrote the SSM written as a single matrix. This matrix M satisfies some special properties: 
1) Lower-triangular (causal)
2) Every submatrix below or on the diagonal is low rank — at most rank N.

These kinds of matrices are called **semiseparable matrices**. 

### 4.3 What is a semiseperable matrix ? ###

A (lower triangular) matrix 𝑀 is N-semiseparable if every submatrix contained in the lower triangular portion (i.e. on or below the diagonal) has rank at most N. We call N the order or rank of the semiseparable matrix.

Why does this matter? Because structured matrices with low-rank off-diagonal blocks have fast algorithms. The paper's central message is this: Different ways of computing SSMs are just different algorithms for multiplying by the semiseparable matrix M.  
The recurrent scan is one algorithm. The naive attention-like matrix multiply is another. The SSD algorithm (coming next) is third (and it is fastest).

### 4.4 Duality Property between SSM and attention ###

Now here's where the scalar-identity restriction on $A_t$​ becomes powerful. If $A_t = a_t \cdot I$, then the product $A_{t:s}^\times = a_t a_{t-1} \cdots a_{s+1}$ is just a scalar. Scalars commute with everything, so we can factor them out of $M_{ts}$​:

$$M_{ts} = C_t^\top \cdot \underbrace{(a_t \cdots a_{s+1})}_{L_{ts}} \cdot B_s = L_{ts} \cdot (C_t^\top B_s)$$

In matrix form, this becomes $M = L \circ C B^T$ where $\circ$ denotes the Hadamard product(elementwise multiplication) and L is the matrix of cumulative products.

For T=4, L would look like: 

$$L = \begin{bmatrix}
1 & & & \\
a_1 & 1 & & \\
a_2a_1 & a_2 & 1 & \\
a_3a_2a_1 & a_3a_2 & a_3 & 1
\end{bmatrix}$$

Each entry $L_{ts}$ denotes *how much does position s still influence position t?* If $a_t$​ values are close to 1, the influence decays slowly. If they're close to 0, it decays fast. Here $a_t$​ is input-dependent — different tokens produce different decay rates.

Now look at the full output: $Y = MX =  (L \circ C B^T)XY$. Compare this to causal linear attention: $Y = (L_{causal} \circ Q K^T)V$, where $L_{\text{causal}}$  is the standard lower-triangular mask. The two expressions are structurally identical if you rename $(C,B,X) \leftrightarrow (Q,K,V)$. The only difference is the mask. In standard linear attention, L is all ones — every past position contributes equally. In SSD, L is a matrix of decaying cumulative products — far positions contribute less.

This is the duality: the same model can be viewed as either a selective SSM recurrence or a masked linear attention. They compute the exact same output.


### 4.5 The SSD algorithm ###

We now have two ways to compute the same model:

+ SSM (recurrent) mode: Compute $h_t = a_t h_{t-1} + B_t x_t$​ step by step. FLOPs scale as O($TN^2$), linear in sequence length. But it's all scalar operations — no tensor cores.
+ Attention (quadratic) mode: Materialize $M = L \circ CB^T$ and compute Y=MX. FLOPs scale as O($T^2 N$), quadratic in sequence length. But it's all matrix multiplications — tensor cores used.

Neither is ideal on its own. The recurrent mode is cheap in FLOPs but hardware-inefficient. The attention mode is hardware-efficient but FLOPs blow up for long sequences. The SSD algorithm combines them. The key idea is a block decomposition of the semiseparable matrix M.

To begin, we partition the matrix 𝑀 into a $\frac{T}{Q} \times \frac{T}{Q}$ grid of submatrices of size Q×Q, for some block size Q. Note that the off-diagonal blocks are low-rank by the defining property of semiseparable matrices.

<img width="626" height="205" alt="image" src="https://github.com/user-attachments/assets/383b6b0b-de1b-4f16-ae35-f0e2c518a3c3" />

This can be illustrated through an example, e.g. for T=9 and decomposing into chunks of length Q=3.The shaded cells are low-rank factorizations of the off-diagonal blocks of the semiseparable matrix.

<img width="568" height="390" alt="image" src="https://github.com/user-attachments/assets/ce720d1d-6307-4a38-9a99-cf742028191c" />

<img width="600" height="320" alt="image" src="https://github.com/user-attachments/assets/1de55181-5367-4790-847c-2c757a514189" />

+ First **Split the Sequence into Chunks**. Let's say our sequence has T=9 tokens and we split into chunks of size Q = 3. So we have 3 chunks:

  + Chunk 0: tokens {0,1,2}
  + Chunk 1: tokens {3,4,5}
  + Chunk 2: tokens {6,7,8}

  Now partition M into a 3×3 grid of 3×3 blocks:

$$
M = \begin{bmatrix}
M^{(0,0)} & 0 & 0 \\
M^{(1,0)} & M^{(1,1)} & 0 \\
M^{(2,0)} & M^{(2,1)} & M^{(2,2)}
\end{bmatrix}
$$

  The upper triangle is all zeros because M is lower triangular (causality — future can't influence past). Now there are two types of blocks. Let's understand each one separately.

+ **The Diagonal Blocks**: $M^(0,0), M^{(1,1)}, M^{(2,2)}$ sit on the diagonal. Let's look at $M^{(1,1)}$ as a concrete example. It covers rows {3,4,5} and columns {3,4,5} of M:

$$
M^{(1,1)} = \begin{bmatrix}
C_3^\top B_3 & 0 & 0 \\
C_4^\top a_4 B_3 & C_4^\top B_4 & 0 \\
C_5^\top a_5 a_4 B_3 & C_5^\top a_5 B_4 & C_5^\top B_5
\end{bmatrix}
$$

This block answers the question: within chunk 1, how does each token influence the others?

$M^{(1,1)}$ is itself a small semiseparable matrix but it is just for 3 tokens instead of 9. So we can compute it using the quadratic attention form:

$$M^{(1,1)} = L^{(1)} \circ C_{3:6} B_{3:6}^\top$$

where $L^{(1)}$ is the small 3×3 decay mask built from $a_3, a_4, a_5$​. This is a tiny matrix multiply and more importantly uses tensor cores, hence fast.

Key insight about diagonal blocks: They only need inputs from within their own chunk. They are completely independent of all other chunks. So all three diagonal blocks can be computed fully in parallel.


+ **The off-diagonal blocks**: Now look at $M^{(2,0)}$. It covers rows {6,7,8} and columns {0,1,2}. Entry $M^{(2,0)}_{ts}$​ is:

$$
M_{ts} = C_t^\top \cdot (a_t \cdots a_{s+1}) \cdot B_s
$$

For example, $$M_{6,0} = C_6^\top \cdot (a_6 a_5 a_4 a_3 a_2 a_1) \cdot B_0$$

These entries span across chunk boundaries. Token 0 (in chunk 0) influencing token 6 (in chunk 2) requires carrying information across two chunk boundaries.

The product $a_t \cdots a_{s+1}$​ for any t in chunk 2 and any s in chunk 0 can be split at the chunk boundary:

$$
a_t \cdots a_{s+1} = \underbrace{(a_t \cdots a_6)}_{\text{chunk 2 part}} \cdot \underbrace{(a_5 a_4 a_3)}_{\text{crossing}} \cdot \underbrace{(a_2 \cdots a_{s+1})}_{\text{chunk 0 part}}
$$

Hence, the whole block factorizes as: 

$$M^{(2,0)} = \underbrace{\begin{bmatrix} 
C_6^\top a_6 \\ 
C_7^\top a_7 a_6 \\ 
C_8^\top a_8 a_7 a_6 
\end{bmatrix}}_{\text{C-block (chunk 2)}} \cdot \underbrace{(a_5 a_4 a_3)}_{\text{boundary scalar}} \cdot \underbrace{\begin{bmatrix} a_2 a_1 B_0 & a_2 B_1 & B_2 \end{bmatrix}}_{\text{B-block (chunk 0)}}$$

A matrix of shape (3×N) times a scalar times a matrix of shape (N×3). So resultant is (NxN). This is the defining property of semiseparable matrices — all off-diagonal blocks are low rank.


1) Step 1 — Intra-chunk outputs (diagonal blocks)
   
   For each chunk j, compute: $Y_j^{\text{intra}} = M^{(j,j)} \cdot X_j$​
    This uses only inputs within the chunk and produces outputs assuming the hidden state entering the chunk is zero. It's a small matrix multiply. All chunks computed in parallel and uses the     Tensor cores.

3) Step 2 — Chunk final states (the B-blocks)

  For each chunk j, compute the final hidden state that chunk would produce if it started from zero:

$$
\text{state}_j = \sum_{s \in \text{chunk } j} (\text{decay from } s \text{ to end of chunk}) \cdot B_s x_s
$$

  This is the B-block column vector from the low-rank factorization. For chunk 0, this is the vector $[a_2 a_1 B_0 x_0 + a_2 B_1 x_1 + B_2 x_2]$ — i.e. the final state of chunk 0 assuming it     started from zero.

3) Step 3 — Pass states across chunks (the boundary scalars)
  Now we need to figure out the true initial state for each chunk, taking into account all the chunks that came before it.

  The true initial state of chunk 1 = (boundary scalar) × (local final state of chunk 0).  
  The true initial state of chunk 2 = (boundary scalar) × (true initial state of chunk 1) + (boundary scalar) × (local final state of chunk 1).
  
  This is itself a recurrence on a sequence of length T/Q instead of T. In our example, 3 chunks instead of 9 tokens. We run a scalar SSM scan on this short sequence.  
  Because this sequence is Q times shorter, the scan is Q times cheaper. Think of it as: "what is the true accumulated hidden state arriving at the start of each chunk, after accounting for everything before it?"

4) Step 4 — State-to-output correction (the C-blocks)

  Now each chunk knows its true initial state $h_{\text{init}}^{(j)}$​ from Step 3. This initial state also contributes to the outputs within the chunk. We need to add that contribution in.

  For each chunk j, compute:

$$
Y_j^{\text{correction}} = (\text{C-block of chunk } j) \cdot h_{\text{init}}^{(j)}
$$

This is the C-block row vector from the low-rank factorization, applied to the initial state. Another batched matrix multiply. All chunks in parallel.
Think of it as: "given that there was already some hidden state arriving at the start of this chunk, how does it affect the outputs?"

Final output: $Y^{\text{intra}} + Y^{\text{correction}}$

The intra-chunk part captures within-chunk interactions. The correction part captures how earlier chunks influence later ones through the hidden state.

**Why This Is Fast**: Steps 1, 2, and 4 are all batched matrix multiplications using Tensor cores. Step 3 is a scan — but on a sequence Q times shorter. For Q=64, it's 64× cheaper than a full scan on the original sequence. In practice it takes a negligible fraction of total time.

The total FLOPs are O($TN^2$) — same as the pure SSM recurrence. But now most of those FLOPs go through tensor cores, which are 16× faster than general arithmetic on modern GPUs. That's where the 2–8× speedup over Mamba-1 comes from.

### 4.6 A Numerical Subtlety Worth Knowing ###

Building the matrix L requires computing cumulative products like $a_t \cdot a_{t-1} \cdots a_{s+1}$. If $a_t \approx 0.9$ and T=2000, you're computing $0.9^{2000} \approx 10^{-91}$. That vanishes to zero in floating point immediately.

The natural fix is to work in log-space. Instead of multiplying, you add logs: $$\log(a_t \cdots a_{s+1}) = \log a_t + \cdots + \log a_{s+1}$$

Then $$L_{ts} = \exp(\text{segsum}(a)_{ts})$$ where segsum is a "segment sum" — the sum of log-a values over a contiguous segment [s+1,t].


### 4.7 The Mamba 2 block ###

<img width="731" height="400" alt="image" src="https://github.com/user-attachments/assets/d8286200-2e8c-4806-88e1-7d60260ba130" />

Mamba 2 architecture is different from Mamba 1's in the following ways:

1) Parallel vs. Sequential Projections: In Mamba-1, the SSM parameters ($A, B, C$) are generated sequentially after $X$ is processed. In Mamba-2, $A, B, C,$ and $X$ are all projected at the exact same time right at the beginning of the block. This is done for Hardware efficiency. Generating them in parallel makes it much easier to split the model across multiple GPUs. It mirrors how Transformers efficiently generate $Q, K,$ and $V$ all at once.
2) Extra Normalization: Mamba-2 adds a new normalization step (the circle labeled "N") right before the final linear projection. This improves model stability. As models scale up in size, this extra normalization keeps the values from exploding or collapsing.


One more important clarification: 

You might think that making matrix A scalar and tying all N state dimensions to the same decay scalar $a_t$​, are we throwing away too much expressivity?

This is okay because: In Mamba-1, the different diagonal values of $A_t$  allowed different parts of the hidden state to forget at different rates. But the $B_t$ and $C_t$​ projections already have full freedom to weight how information enters and exits the state. The scalar $a_t$​ controls the *overall* forgetting rate — whether this timestep matters at all. The paper's ablations confirm this: the scalar restriction doesn't hurt quality, especially when N is increased.

And increasing N is exactly what SSD enables. Mamba-1 was limited to N=16 because the scan cost scaled with N. SSD is dominated by matrix multiplications, so larger N can be used. Mamba-2 uses N=64 or 128 or even 156. More state capacity means the model can remember more — and that turns out to matter a lot.





---

## 6. Benchmarking Mamba-1, Mamba-2 and Transformers on HellaSwag ##

To make things practical, I ran zero-shot evaluations on three benchmarks using the lm-evaluation-harness, same like in the Mamba paper. I checked Mamba-1, Mamba-2, and also two transformer baselines.

I could not benchmark Mamba-3 because its pre-trained weights have not been released yet.

### 6.1 Benchmarks Used ###

+ **HellaSwag**: A commonsense reasoning benchmark where the model must pick the most plausible continuation of a sentence from four choices. It tests whether the model understands everyday situations and physical common sense.
+ **LAMBADA OpenAI**: Tests whether the model can predict the last word of a passage that requires understanding the broader context of the whole paragraph — not just the last few words. It measures long-range language coherence. Lower perplexity and higher accuracy are both better here.
+ **ARC Challenge**: Comprises of grade-school science questions specifically designed to be hard. Questions that simple word-matching or retrieval cannot answer. It requires reasoning ability.


### 6.2 Setup ###

All models were evaluated zero-shot (n-shot = 0), meaning no examples were provided before testing. The primary metric for HellaSwag and ARC Challenge is acc_norm (length-normalized accuracy) which corrects for the model's natural bias toward shorter answers. For LAMBADA, both perplexity and acc are reported.

Models evaluated:

+ Mamba-1 1.4B — the selective SSM from Section 3
+ Mamba-2 1.3B — the SSD-based architecture from Section 4
+ Pythia 1.4B — a transformer trained on the same dataset as Mamba (The Pile), making it the cleanest comparison.
+ TinyLlama 1.1B — a modern transformer using the LLaMA architecture with RoPE, SwiGLU, and grouped query attention

### 6.3 Results ###

<img width="680" height="344" alt="image" src="https://github.com/user-attachments/assets/db536e2a-818d-4f2d-b969-b18107a57a8d" />



### 6.4 Observations ###

**HellaSwag**

1) Mamba-1 matches transformers at the same parameter count. Mamba-1 1.4B and TinyLlama 1.1B score 0.5913 and 0.5920 respectively, which is almost identical. This is significant because TinyLlama uses years of accumulated transformer engineering (RoPE, SwiGLU, grouped query attention) while Mamba-1 uses no attention mechanism at all.
2) Mamba-1 outperforms Pythia at the same size. Both are 1.4B models trained on The Pile dataset, making this the cleanest comparison. Mamba-1 scores 0.5913 against Pythia's 0.5197. Hence, Mamba-1 scores somewhat better than its corresponding transformer rival Pythia.
3) Mamba-2 achieves the highest score (0.5994), again showing its architecture benefits more from scale.

**LAMBADA OpenAI**

1) Both Mamba-1 1.4B and Mamba-2 1.3B achieve perplexity around 5.0, while Pythia sits at 6.09 and TinyLlama at 6.93. This implies Mamba is clearly better at long-range language coherence.
2) On accuracy, Mamba-2 1.3B scores highest at 0.6555, followed by Mamba-1 1.4B at 0.6493. Both clearly outperform Pythia (0.6158) and TinyLlama (0.5882).
3) This result makes sense given Mamba's architecture. LAMBADA rewards models that can maintain context over long passages — exactly what SSMs with their efficient state compression are designed to do.

**ARC Challenge**

1) ARC Challenge is the hardest benchmark of the three. Thus, all models are in the 0.28–0.33 range on acc_norm, showing that genuine scientific reasoning remains difficult at this scale of parameters.
2) Mamba-2 1.3B scores highest at 0.3319, closely followed by Mamba-1 1.4B at 0.3294. Both outperform Pythia (0.2833) and TinyLlama (0.3012).
3) The fact that both Mamba models beat TinyLlama here is notable — TinyLlama uses a more modern transformer architecture, yet Mamba still edges ahead on reasoning tasks.

**What These Numbers Mean**: 

+ On HellaSwag, random chance is 0.25 (choosing out of four options), human performance is around 0.95, and models like LLaMA-3 8B reach approximately 0.82. Our models scoring ~0.59-0.60 shows they have learned meaningful language representations, but a large gap to human-level reasoning remains — one that generally closes with scale.
+ On LAMBADA, a perplexity of ~5.0 for Mamba models is strong at this parameter count. For reference, GPT-2 1.5B scores around 8.6. So Mamba at 5.0 with a fewer parameters is performing well on this.
+ On ARC Challenge, scores around 0.30-0.33 reflect how hard this benchmark is. Random chance is 0.25 and even LLaMA-3 8B only reaches ~0.57. ARC requires reasoning that only emerges reliably at much larger scales.
Across all three, Mamba consistently sits at the top of the 1-1.4B range, which is the key takeaway.


#### Clarification about how accuracy is calculated ####

When lm_eval runs HellaSwag, it scores each of the 4 candidate completions by computing the log-probability of that completion given the context. The candidate with the highest log-probability is picked as the answer. Formula for log-probability

$$\log P(\text{completion}) = \sum_{i=1}^{n} \log P(\text{token}_i)$$

Each individual token probability is less than 1, so its log is negative. This means the more tokens a completion has, the more negative its total score becomes because of adding more negative numbers.

acc_norm divides the total log-probability by the number of tokens:

$$\text{acc-norm} = \frac{\log P(\text{completion})}{n}$$

This gives a per-token average, putting all candidates on equal level regardless of length. Hence it is used as the standard metric for HellaSwag.

  
The more interesting finding here is the architectural parity. An SSM with linear-time complexity, no attention, and O(1) inference cost matches a transformer with quadratic attention on a standard reasoning benchmark. That is the core promise of the Mamba line of work — and at least at this scale, it holds up.

## 7 References: ##

+ HiPPO: https://arxiv.org/abs/2008.07669
+ S4: https://arxiv.org/abs/2111.00396
+ Mamba1: https://arxiv.org/abs/2312.00752
+ Mamba2: https://arxiv.org/abs/2405.21060
+ Mamba repo: https://github.com/mamba-org/mamba
+ https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state
+ https://tridao.me/blog/
+ https://www.youtube.com/watch?v=8Q_tqwpTpVU
