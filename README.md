# State Spaces & Mamba
This is a blog on State Spaces and Mamba

## 1. Introduction: Sequence Modelling ##

Sequence modelling is a task to map an input sequence x(t), to an output sequence y(t). The input signal could be continuous (like in case of audio) or discrete (like in case of text). Continuous input sequence gets mapped to continuous output sequence and discrete input sequence to a discrete output sequence.

Let's see some of the existing methods for sequence modelling:

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

Based on the benfits and shortcomings of the above models, we can deduce that an ideal model could: 
1. Parallelize the training (like the Transformer) and  scale linearly to long sequences (with a computation/memory cost of O(N) like the RNN) 
2. Can inference each token with a constant computation/memory cost (O(1) like the RNN)

With this in mind, let's explore State Space Models:

## 2 Background: State Spaces ##

Our goal is the efficient modeling of long sequences. To do this, we are going to build a new neural network layer based on State Space Models. 

### 2.1 State Space Models: A Continuous-time Latent State Model ###

The state space model is defined by the following equation: 

$$ h'(t) = Ah(t)+Bx(t) $$
$$ y(t) = Ch(t)+Dx(t) $$

It maps a 1-D input signal x(t) to an N-D latent state h(t) before projecting to a 1-D output signal y(t).

Our goal is to simply use the SSM as a black-box representation in a deep sequence model, where B,C,D are parameters learned by gradient descent and A is a special HiPPO matrix(covered in upcoming section). For the remainder of this blog, we will omit the parameter D for exposition because the term $Dx$ can be viewed as a skip connection and is easy to compute.

This state space model is linear and time invariant. Linear because the relationships in the expressions above are linear, and time invariant becuase A,B,C,D do not depend on time(they are fixed).

Note: for now consider A, B, C, D, x(t), h(t) and y(t) to be numbers, not vectors. Later we will extend our analysis to vectors.

**Significance of Matrix A**: The A matrix in the state space model can be thought of as a matrix that “captures” information from the previous state to build the new state. It determines how this information is propagated over time. Because of this, its structure must be designed carefully—otherwise, it may fail to effectively retain the history of past inputs, which is essential for generating accurate future outputs. To make the A matrix behave well, the authors chose to use the HIPPO theory. Let’s see how it works!

###2.2 Adressing long range dependencies using HiPPO###

HiPPO specifies a class of certain matrices $A ∈ R^N×N$ that when incorporated into SSM, allows the state h(t) to memorize the history of the input x(t). The most important matrix in this class is defined by below equation, which we will call the HiPPO matrix: 
<img width="1221" height="219" alt="image" src="https://github.com/user-attachments/assets/71603f1a-5854-402f-861a-524b829d024b" />

For those not intersted to know idea behind HiPPO and directly use the above A matrix can skip to section 2.3.

At every time step, a sequence model must summarize everything seen so far, not just the last few tokens. And it must do this with a fixed-size vector. This is a compression problem. HiPPO reframes memory as: at every time t, find the best polynomial approximation of the input history f(x) for x ≤ t. 

# State Spaces & Mamba
This is a blog on State Spaces and Mamba

## 1. Introduction: Sequence Modelling ##

Sequence modelling is a task to map an input sequence x(t), to an output sequence y(t). The input signal could be continuous (like in case of audio) or discrete (like in case of text). Continuous input sequence gets mapped to continuous output sequence and discrete input sequence to a discrete output sequence.

Let's see some of the existing methods for sequence modelling:

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

Based on the benfits and shortcomings of the above models, we can deduce that an ideal model could: 
1. Parallelize the training (like the Transformer) and  scale linearly to long sequences (with a computation/memory cost of O(N) like the RNN) 
2. Can inference each token with a constant computation/memory cost (O(1) like the RNN)

With this in mind, let's explore State Space Models:

## 2 Background: State Spaces ##

Our goal is the efficient modeling of long sequences. To do this, we are going to build a new neural network layer based on State Space Models. 

### 2.1 State Space Models: A Continuous-time Latent State Model ###

The state space model is defined by the following equation: 

$$ h'(t) = Ah(t)+Bx(t) $$
$$ y(t) = Ch(t)+Dx(t) $$

It maps a 1-D input signal x(t) to an N-D latent state h(t) before projecting to a 1-D output signal y(t).

Our goal is to simply use the SSM as a black-box representation in a deep sequence model, where B,C,D are parameters learned by gradient descent and A is a special HiPPO matrix(covered in upcoming section). For the remainder of this blog, we will omit the parameter D for exposition because the term $Dx$ can be viewed as a skip connection and is easy to compute.

This state space model is linear and time invariant. Linear because the relationships in the expressions above are linear, and time invariant becuase A,B,C,D do not depend on time(they are fixed).

Note: for now consider A, B, C, D, x(t), h(t) and y(t) to be numbers, not vectors. Later we will extend our analysis to vectors.

**Significance of Matrix A**: The A matrix in the state space model can be thought of as a matrix that “captures” information from the previous state to build the new state. It determines how this information is propagated over time. Because of this, its structure must be designed carefully—otherwise, it may fail to effectively retain the history of past inputs, which is essential for generating accurate future outputs. To make the A matrix behave well, the authors chose to use the HIPPO theory. Let’s see how it works!


###2.2 Adressing long range dependencies using HiPPO###

HiPPO specifies a class of certain matrices $A ∈ R^N×N$ that when incorporated into SSM, allows the state h(t) to memorize the history of the input x(t). The most important matrix in this class is defined by below equation, which we will call the HiPPO matrix: 
<img width="1221" height="219" alt="image" src="https://github.com/user-attachments/assets/71603f1a-5854-402f-861a-524b829d024b" />

For those not intersted to know idea behind HiPPO and directly use the above A matrix can skip to section 2.3.

At every time step, a sequence model must summarize everything seen so far, not just the last few tokens. And it must do this with a fixed-size vector. This is a compression problem. HiPPO reframes memory as: at every time t, find the best polynomial approximation of the input history f(x) for x ≤ t. 
Why polynomials? They're a universal basis. They're well-understood. And crucially their optimal coefficients can be computed online as new data arrives. 



### 2.3 Discretization ###

Usually we never work with continuous signals, but always with discrete ones (like text), so how can we produce outputs 𝑦(𝑡) for a discrete signal? Moreover, solving the ODE analytically can be difficult and cumbersome. So, we first need to discretize our system!

Text, audio waveform, or sensor readings aren't a smooth continuous signal — they're a discrete sequence of tokens or samples: $u_0, u_1, u_2, ...$ arriving at fixed intervals. To run an SSM on real data, we need to *discretize* it — convert those continuous matrices (A,B,C) into discrete counterparts $(\bar{A}, \bar{B}, \bar{C})$  that can step through a sequence one token at a time.

**The Step Size $\Delta$**: Represents the time interval between two consecutive inputs. Conceptually, think of each discrete input $u_k = u(k\Delta)$​ as a *sample* of an underlying continuous signal at time t=kΔ.
A small Δ: means sampling densely — high resolution. A large Δ means coarser steps — the model "skips" more of the underlying dynamics between tokens.

<img width="720" height="233" alt="image" src="https://github.com/user-attachments/assets/925eff2e-99a6-4a83-a50c-23a15836c40b" />

The discretization method used is called Bilinear method: it approximates the continuous derivative $h'(t)$ using the trapezoidal rule — averaging the state at the beginning and end of the interval. We will not go into the derivation of it but the final matrices after discretization will be:

<img width="575" height="155" alt="image" src="https://github.com/user-attachments/assets/360857a7-29f7-4946-a103-246b40fbaa13" />

So, SSM equation ius now a sequence-to-sequence map $u_k → y_k$ instead of function-to-function. Moreover the state equation is now a recurrence in $x_k$, allowing the discrete SSM to be computed like an RNN. $x_k ∈ R^N$ can be viewed as a hidden state with transition matrix A.


Why polynomials? They're a universal basis. They're well-understood. And crucially their optimal coefficients c(t) can be computed online as new data arrives via a linear ODE. 


<img width="996" height="551" alt="image" src="https://github.com/user-attachments/assets/1f02c940-df91-42d2-93a6-41033621a057" />
The above diagram summarizes the whole idea of HiPPO.

**The HiPPO Framework: Online Function Approximation**: Given a measure μ(t) that weights the importance of the past (e.g., "pay more attention to recent history"), HiPPO finds the polynomial g(t) of degree < N that best approximates f up to time t in the L² sense: 

$g(t)=\arg\min_{g \in \mathcal{G}} \| f_{\leq t} - g \|_{L^2(\mu^{(t)})}$ 
​
The brilliant result is that the N coefficients c(t) of this best approximation evolve according to a linear ODE:

$\frac{d}{dt} c(t) = A(t) \cdot c(t) + B(t) \cdot f(t)$
This is exactly the SSM structure from Section 2.1! The A and B matrices here are derived from first principles from the choice of measure, not learned randomly. The state c(t) ∈ ℝᴺ is our hidden state h(t), and it provably encodes an optimal polynomial summary of the entire input history.

### 2.3 Discretization ###

Usually we never work with continuous signals, but always with discrete ones (like text), so how can we produce outputs 𝑦(𝑡) for a discrete signal? Moreover, solving the ODE analytically can be difficult and cumbersome. So, we first need to discretize our system!

Text, audio waveform, or sensor readings aren't a smooth continuous signal — they're a discrete sequence of tokens or samples: $u_0, u_1, u_2, ...$ arriving at fixed intervals. To run an SSM on real data, we need to *discretize* it — convert those continuous matrices (A,B,C) into discrete counterparts $(\bar{A}, \bar{B}, \bar{C})$  that can step through a sequence one token at a time.

**The Step Size $\Delta$**: Represents the time interval between two consecutive inputs. Conceptually, think of each discrete input $u_k = u(k\Delta)$​ as a *sample* of an underlying continuous signal at time t=kΔ.
A small Δ: means sampling densely — high resolution. A large Δ means coarser steps — the model "skips" more of the underlying dynamics between tokens.


The discretization method used is called Bilinear method: it approximates the continuous derivative $h'(t)$ using the trapezoidal rule — averaging the state at the beginning and end of the interval. We will not go into the derivation of it but the final matrices after discretization will be:

<img width="575" height="155" alt="image" src="https://github.com/user-attachments/assets/360857a7-29f7-4946-a103-246b40fbaa13" />

So, SSM equation ius now a sequence-to-sequence map $u_k → y_k$ instead of function-to-function. Moreover the state equation is now a recurrence in $x_k$, allowing the discrete SSM to be computed like an RNN. $x_k ∈ R^N$ can be viewed as a hidden state with transition matrix A.
