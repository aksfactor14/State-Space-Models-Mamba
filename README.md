# State Spaces & Mamba
This is a blog on State Spaces and Mamba

## 1. Introduction: Sequence Modelling and Existing Methods ##

Sequence modelling is a task to map an input sequence x(t), to an output sequence y(t). The input signal could be continuous (like in case of audio) or discrete (like in case of text). Continuous input sequence gets mapped to continuous output sequence and discrete input sequence to a discrete output sequence.

**Why study MAMBA and SSMs**: Before we dive into State Space Models, it helps to understand why they exist. To do that, we need to look at the two dominant approaches to sequence modeling — Transformers and RNNs — and understand where each one breaks down.


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
      <td>❌ No — O(N)</td>
      <td>✅ Yes</td>
      <td>✅ Yes — O(N²)</td>
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

**Transformers**: Transformers are the SOTA of modern AI. Their self-attention mechanism is powerful because every token in a sequence can directly attend to every other token — no information gets lost through compression. 

**The problem**: The power of self attention comes with a cost inherited into the architecture itself. At training time, self-attention is O($N^2$) in both time and memory. At inference, KV-caching reduces this to O(N) per token — but memory still grows linearly with context length.

For a sequence of length N, attention computes a similarity score between every pair of tokens. Double the sequence length, and you quadruple the compute. At 10K tokens you're already straining memory. At 100K tokens, you're in trouble.

**Aren't RNNs the Answer?**: RNNs(and LSTMs or GRU) compress the entire past into a hidden state $h_t$​, updated step-by-step:

$h_t = f(h_{t-1},x_t)$

This is elegant. A fixed-size memory that gets updated as the sequence progresses and inference is O(1) per token.

Problems with RNNs:
1. Vanishing gradients. When training through long sequences, gradients must flow backwards through hundreds of steps. They almost always vanish (or explode) before reaching early tokens. 
2. No parallelism during training. An RNN is inherently sequential — step t depends on step $t_{-1}$ which depends on step $t_{-2}$. So these can't be computed in parallel


Based on the benfits and shortcomings of the above models, we can deduce that an ideal model could: 
1. Parallelize the training (like the Transformer) and  scale linearly to long sequences (with a computation/memory cost of O(N) like the RNN) 
2. Can inference each token with a constant computation/memory cost (O(1) like the RNN)

With this in mind, let's explore State Space Models:

## 2. Background: State Spaces ##

Our goal is the efficient modeling of long sequences. To do this, we are going to build a new neural network layer based on State Space Models. 

### 2.1 State Space Models: A Continuous-time Latent State Model ###

The state space model is defined by the following equation: 

$$ h'(t) = Ah(t)+Bx(t) $$
$$ y(t) = Ch(t)+Dx(t) $$

It maps a 1-D input signal x(t) to an N-D latent state h(t) before projecting to a 1-D output signal y(t).

Our goal is to simply use the SSM as a black-box representation in a deep sequence model, where B,C,D are parameters learned by gradient descent and A is a special HiPPO matrix(covered in upcoming section). For the remainder of this blog, we will omit the parameter D for exposition because the term $Dx$ can be viewed as a skip connection and is easy to compute.

This state space model is linear and time invariant. Linear because the relationships in the expressions above are linear, and time invariant becuase A,B,C,D do not depend on time(they are fixed).

Note: for now consider A, B, C, D, x(t), h(t) and y(t) to be numbers, not vectors. Later we will extend our analysis to vectors.

**Significance of Matrix A**: The A matrix in the SSM “captures” information from the previous state to build the new state. It determines how this information is propagated over time. Because of this, its structure must be designed carefully—otherwise, it may fail to effectively retain the history of past inputs. To make the A matrix behave well, the authors chose to use the HIPPO theory. Let’s see how it works!


### 2.2 Adressing long range dependencies using HiPPO ###

HiPPO specifies a class of certain matrices $A ∈ R^N×N$ that when incorporated into SSM, allows the state h(t) to memorize the history of the input x(t). The most important matrix in this class is defined by below equation, which we will call the HiPPO matrix: 
<img width="610" height="110" alt="image" src="https://github.com/user-attachments/assets/71603f1a-5854-402f-861a-524b829d024b" />

For those not intersted to know idea behind HiPPO and directly use the above A matrix can skip to section 2.3.

At every time step, a sequence model must summarize everything seen so far, not just the last few tokens. And it must do this with a fixed-size vector. This is a compression problem. HiPPO reframes memory as: at every time t, find the best polynomial approximation of the input history f(x) for x ≤ t. 

<img width="500" height="225" alt="image" src="https://github.com/user-attachments/assets/1f02c940-df91-42d2-93a6-41033621a057" />

The above diagram summarizes the whole idea of HiPPO.

**The HiPPO Framework: Online Function Approximation**: Given a measure μ(t) that weights the importance of the past (e.g., "pay more attention to recent history"), HiPPO finds the polynomial g(t) of degree < N that best approximates f up to time t in the L² sense: 

$g(t)=\arg\min_{g \in \mathcal{G}} \| f_{\leq t} - g \|_{L^2(\mu^{(t)})}$ 
​
The result is that the N coefficients c(t) of this approximation evolve according to a linear ODE:

$$ \frac{d}{dt} c(t) = A(t) \cdot c(t) + B(t) \cdot f(t) $$

The A and B matrices here are derived from first principles from the choice of measure, not learned randomly. The state c(t) ∈ ℝᴺ is our hidden state h(t), and it encodes an optimal polynomial summary of the entire input history.

**The Three Measure Families**: The choice of measure μ(t) determines what kind of history gets remembered. The paper proposes three families, each with different tradeoffs:

<img width="600" height="377" alt="image" src="https://github.com/user-attachments/assets/524e236d-1f93-4772-92b2-dd97d4a556d1" />


1) LegT (Translated Legendre) assigns uniform weight to the most recent window [t−θ, t]. This is like a sliding window — it eventually forgets old history as the window slides forward. The window size θ is a hyperparameter which needs to be manually selected, which must match the sequence length. If you mis-specify it, performance drops dramatically.

2) LagT (Translated Laguerre) uses an exponentially decaying measure — it remembers all history but gives exponentially less weight to older inputs. It's smoother than LegT but still requires a step-size hyperparameter Δt.

3) LegS (Scaled Legendre) is the key innovation. It assigns uniform weight over the entire history [0, t], and the window grows with time. This gives it two remarkable properties: it requires no timescale hyperparameter, and it is provably invariant to changes in input timescale (if you speed up or slow down the input signal, the coefficients simply shift accordingly). This is the measure used in the SSMs that power models like Mamba.

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

So to summarize: The A matrix is the HiPPO-LegS A matrix, derived from first principles as the optimal online polynomial compression of the input signal's history. This gives the hidden state h(t) a principled and theoretically grounded memory mechanism. We now have a continuous-time SSM with a well-designed A matrix. 

The next challenge is making this practically usable on real discrete data like text or audio. That requires converting these continuous equations into a discrete sequence-to-sequence map — which is exactly what discretization in Section 2.3 achieves.




### 2.3 Discretization ###

Usually we never work with continuous signals, but always with discrete ones (like text), so how can we produce outputs 𝑦(𝑡) for a discrete signal? Moreover, solving the ODE analytically can be difficult and cumbersome. So, we first need to discretize our system!

Text, audio waveform, or sensor readings aren't a smooth continuous signal — they're a discrete sequence of tokens or samples: $u_0, u_1, u_2, ...$ arriving at fixed intervals. To run an SSM on real data, we need to *discretize* it — convert those continuous matrices (A,B,C) into discrete counterparts $(\bar{A}, \bar{B}, \bar{C})$  that can step through a sequence one token at a time.

**The Step Size $\Delta$**: Represents the time interval between two consecutive inputs. Conceptually, think of each discrete input $u_k = u(k\Delta)$​ as a *sample* of an underlying continuous signal at time t=kΔ.
A small Δ: means sampling densely — high resolution. A large Δ means coarser steps — the model "skips" more of the underlying dynamics between tokens.

<img width="480" height="155" alt="image" src="https://github.com/user-attachments/assets/925eff2e-99a6-4a83-a50c-23a15836c40b" />

The discretization method used is called Bilinear method. We will not go into the derivation but the final matrices after discretization will be:

<img width="575" height="155" alt="image" src="https://github.com/user-attachments/assets/360857a7-29f7-4946-a103-246b40fbaa13" />

just to avoid confusion later, I would like to make it clear that S4 paper used te bilinear discretization method and the Mamba paper used the Zero Order Hold (ZOH) for discretization.

<img width="1039" height="140" alt="image" src="https://github.com/user-attachments/assets/844e2a9c-7811-4440-9682-cb87c45d5624" />


Now the result of discretization is that the SSM equation ius now a sequence-to-sequence map $u_k → y_k$ instead of function-to-function. Moreover the state equation is now a recurrence in $x_k$, allowing the discrete SSM to be computed like an RNN. $x_k ∈ R^N$ can be viewed as a hidden state with transition matrix A.


### 2.4 Computing the SSM ###

**The Recurrent View (Great for Inference, Terrible for Training)**

At any time step $t$, the hidden state $h_t$ and output $y_t$ are calculated as: 

$$h_t = \mathbf{\bar{A}}h_{t-1} + \mathbf{\bar{B}}x_t$$

$$y_t = \mathbf{C}h_t$$

During inference, this is beautiful. To generate the next token, you only need the current input $x_t$ and the previous compressed state $h_{t-1}$. It requires constant $O(1)$ memory and time. But during training, this is terrible as you have o compute all tokens sequentially. You cannot compute $h_3$ until you finish computing $h_2$. It completely wastes the massive parallel compute power of modern GPUs.
<img width="656" height="295" alt="image" src="https://github.com/user-attachments/assets/dcfccd72-df64-470d-a68e-9d6b425f041c" />


**The Convolutional View (The Parallel Training Trick)**

The recurrent SSM is not practical for training on GPUs due to its sequentiality. Instead, as SSM matrices are LTI, we can directly use convolutions.

Assume our initial state $h_{-1} = 0$. Let's manually calculate the first few outputs:

$h_{-1} = 0$

$$h_0 = \mathbf{\bar{B}}x_0 \implies y_0 = \mathbf{C\bar{B}}x_0$$

$$h_1 = \mathbf{\bar{A}}h_0 + \mathbf{\bar{B}}x_1 = \mathbf{\bar{A}\bar{B}}x_0 + \mathbf{\bar{B}}x_1 \implies y_1 = \mathbf{C\bar{A}\bar{B}}x_0 + \mathbf{C\bar{B}}x_1$$

$$h_2 = \mathbf{\bar{A}}h_1 + \mathbf{\bar{B}}x_2 \implies y_2 = \mathbf{C\bar{A}^2\bar{B}}x_0 + \mathbf{C\bar{A}\bar{B}}x_1 + \mathbf{C\bar{B}}x_2$$

The output $y_k$ at any step is just a linear combination of all past inputs multiplied by a predictable set of weights.

We can group these weights into a single, massive vector called the SSM Kernel ($\mathbf{\bar{K}}$) 

$$\mathbf{\bar{K}} = (\mathbf{C\bar{B}}, \mathbf{C\bar{A}\bar{B}}, \mathbf{C\bar{A}^2\bar{B}}, \dots, \mathbf{C\bar{A}^L\bar{B}})$$

$$y = x * \mathbf{\bar{K}}$$

<img width="650" height="250" alt="image" src="https://github.com/user-attachments/assets/dafe7cfc-29aa-4ae0-b4e6-09b8ed7e16ad" />

Because this kernel is entirely predictable, we completely bypass the sequential step-by-step calculation. We can just take our entire input sequence $\mathbf{x}$ and apply a standard mathematical convolution:

### 2.5 The LTI trap ###

This convolutional trick is why S4 can train lightning-fast using FFTs across the whole sequence at once. However, the above trick only works because the matrices $\mathbf{\bar{A}}$, $\mathbf{\bar{B}}$, and $\mathbf{C}$ are LTI. They do not change based on the input. 

If $\mathbf{\bar{B}}$ changed its value every time it saw a different word, you couldn't pre-compute the kernel $\mathbf{\bar{K}}$. The convolution trick would shatter. This is the exact bottleneck Mamba had to solve: How do we make the model dynamic and content-aware (breaking the LTI rule) without losing the ability to train fast on GPUs?

**Motivating Mamba**: 

The authors of the Mamba paper describe two tasks on which SSM or the S4 do not perform well.

1) Selective Copying:
   <img width="450" height="170" alt="image" src="https://github.com/user-attachments/assets/8e238e41-9b52-4608-b177-8927cae8861b" />
2) Induction Heads:
   <img width="440" height="225" alt="image" src="https://github.com/user-attachments/assets/ab8ac6d3-9676-4c49-9453-0484b4335d0f" />

These tasks reveal the failure mode of LTI models. From the recurrent view, their constant dynamics (e.g. the (A, B) transitions) cannot let them select the correct information from their context, or affect the hidden state passed along the sequence an in input-dependent way. From the convolutional view, it is known that global
convolutions can solve the vanilla Copying task because it only requires time-awareness, but that they have difficulty with the Selective Copying task because of lack of content-awareness. More concretely, the spacing between inputs-to-outputs is varying and cannot be modeled by static convolution kernels.

These shortcomings led to the development of Mamba....

+ Before proceeding: If you want to deep dive into the technical details on how to calculate the HiPPO matrix and build a S4 model yourself, you may find this helpful https://srush.github.io/annotated-s4/





## 3. Mamba1: Linear-Time Sequence Modeling with Selective State Spaces ##

Some modifications that can be made to S4 to improve its content based reasoning: 
1) Simply letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token.
2) Even though this change prevents the use of efficient convolutions, the authors of the mamba paper propose a hardware-aware parallel algorithm in recurrent mode.
   Then we can integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba).






