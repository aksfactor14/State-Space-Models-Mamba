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

Usually we never work with continuous signals, but always with discrete ones (like text), so how can we produce outputs 𝑦(𝑡) for a discrete signal? Moreover, solving the ODE analytically can be difficult and cumbersome. So, we first need to discretize our system!

### 2.2 Discretization ###


      
