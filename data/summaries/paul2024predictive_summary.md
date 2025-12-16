# On Predictive planning and counterfactual learning in active inference

**Authors:** Aswin Paul, Takuya Isomura, Adeel Razi

**Year:** 2024

**Source:** arxiv

**Venue:** N/A

**DOI:** 10.3390/e26060484

**PDF:** [paul2024predictive.pdf](../pdfs/paul2024predictive.pdf)

**Generated:** 2025-12-15 11:29:38

**Validation Status:** ✓ Accepted
**Quality Score:** 0.80


## Classification

- **Category**: Core Theory Math
- **Domain**: N/A
- **Confidence**: 0.95
- **Reasoning**: The paper's core contribution lies in developing a theoretical framework for intelligent behavior – active inference – based on predictive planning and counterfactual learning. It delves into the mathematical foundations of this framework, exploring concepts like variational free energy, Bayesian inference, and the trade-off between data complexity. The paper doesn't focus on implementing a specific tool or applying it to a particular domain but rather on establishing the theoretical underpinnings of the approach.

---

Okay, here’s a summary of the paper “On Predictive planning and counterfactual learning in active inference” by Paul et al. (2024).### OverviewThis paper investigates two decision-making schemes within the active inference framework: a planning-based approach and a counterfactual learning scheme. The authors aim to develop a hybrid model that balances these strategies to facilitate adaptive decision-making in complex environments, as highlighted by the rapid advancement of artificial intelligence and the need for sophisticated behavioural foundations. The core of the paper lies in demonstrating the feasibility and effectiveness of this mixed model, showcasing its ability to navigate challenges encountered in a mutating grid-world scenario.### MethodologyThe authors establish a theoretical foundation for active inference, emphasizing the importance of maximizing model evidence to perceive, learn, and make decisions. They adopt a partially observable Markov decision process (POMDP) based generative model, a universal framework to model discrete state-space environments. Specifically, they detail the two decision-making schemes: the planning-based scheme, where an agent predicts possible outcomes and makes decisions to attain states and observations that minimise expected free energy (EFE), and the counterfactual learning (CL) scheme, where the agent learns a state-action mapping by accumulating a measure of ‘risk’ over time. They formally define the generative model and the free energy function, which is central to the active inference framework. The authors also quantify the computational complexity associated with the planning-based scheme, which is linearly dependent on the planning depth (T).### ResultsThe primary results demonstrate the feasibility and effectiveness of the proposed hybrid model in a mutating grid-world scenario. The authors show that the agent can learn to navigate the grid within the first300 episodes, even when the environment mutates to a more challenging version with half the acceptable range for both the pole and cart position. Specifically, the agent achieves a successful navigation rate of approximately80% in the mutated grid, compared to a lower rate in the original grid. The authors further observe that the risk parameter (Γt) tends to zero when the easy grid is presented and solved, and it increases when the environment mutates. The improved performance of the CL method agent after mutation warrants additional investigation, but can be attributed to the increased feedback frequency due to the increased failure rate after mutation.### FindingsThe authors’ key findings are:1) The hybrid model effectively balances planning and counterfactual learning, enabling adaptive decision-making;2) The agent can learn to navigate complex environments, even when they undergo mutations;3) The agent’s performance is dependent on the planning depth (T), with a higher planning depth leading to faster learning and improved performance; and4) The computational complexity of the planning-based scheme is linearly dependent on the planning depth (T).### DiscussionThe authors explain that an additional advantage of the mixed model proposed (and the POMDP-based generative models) is that they can probe the model parameters to understand the basis of intelligent behaviour demonstrated by agents through the lens of active inference. Models that rely on artificial neural networks (ANNs) to scale up the models 

Fountas et al. [2020] have limited explainability regarding how agents make decisions, especially when faced with uncertainty.

In summary, this paper provides a valuable contribution to the field of active inference by demonstrating a practical approach to decision-making in complex environments.
