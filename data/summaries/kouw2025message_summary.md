# Message passing-based inference in an autoregressive active inference agent

**Authors:** Wouter M. Kouw, Tim N. Nisslbeck, Wouter L. N. Nuijten

**Year:** 2025

**Source:** arxiv

**Venue:** N/A

**DOI:** N/A

**PDF:** [kouw2025message.pdf](../pdfs/kouw2025message.pdf)

**Generated:** 2025-12-15 12:11:16

**Validation Status:** ✓ Accepted
**Quality Score:** 0.80


## Classification

- **Category**: Core Theory Math
- **Domain**: N/A
- **Confidence**: 0.95
- **Reasoning**: The paper presents an autoregressive active inference agent based on message passing on a factor graph. The core contribution lies in the derivation of the expected free energy minimization and the formulation of the planning model as a factor graph. The paper focuses on the theoretical foundations of active inference and the mathematical framework for implementing it, rather than a specific application or tool.

---

Okay, here’s a summary of the paper, adhering to all the provided instructions and constraints.### OverviewThis paper introduces an autoregressive active inference agent for robot navigation, leveraging message passing on a factor graph. The agent is designed to learn and execute actions based on predictive uncertainty, offering a more robust approach compared to traditional optimal controllers. The authors demonstrate the agent’s ability to explore and exploit in a continuous-valued observation space, achieving successful navigation while learning a better model of the robot’s dynamics. The key innovation lies in the agent’s ability to modulate actions based on predictive uncertainty, arriving at the goal later but with a more accurate understanding of the system.### MethodologyThe core of the agent is built on an autoregressive model, meaning that the system output at time *k* is predicted from the system input *u*<sub>k</sub>, *M*<sub>u</sub> previous system inputs *¯u*<sub>k</sub>, and *M*<sub>y</sub> previous system outputs *¯y*<sub>k</sub>. The model is defined by a Gaussian likelihood function, p(y<sub>k</sub> |Θ, u<sub>k</sub>,¯u<sub>k</sub>,¯y<sub>k</sub>) = N(y<sub>k</sub> | A<sup>T</sup>x<sub>k</sub>, W<sup>-1</sup>), where A is a regression coefficient matrix and W is a precision matrix. The authors utilize a multivariate autoregressive model with continuous-valued observations and bounded continuous-valued actions. The agent learns through Bayesian filtering, updating parameter beliefs given observations (Eq.7). The filtering process relies on message passing on a factor graph (Figure1), where messages are exchanged between nodes representing observations, parameters, and control inputs. The agent uses a T-distribution to represent the posterior predictive distribution, which is crucial for evaluating the agent’s performance. The agent’s planning model is built on a factor graph with marginal distribution updates based on messages passed along the graph (Figure3). The agent’s expected free energy is minimized using a variational approach, resulting in a Gaussian distribution for the control inputs.### ResultsThe authors validate the proposed design on a robot navigation task, comparing the agent to an adaptive model-predictive controller. The agent consistently scores a smaller free energy than the model-predictive controller, demonstrating its ability to accurately predict its next observation. The agent successfully navigated to the goal position, achieving a better model of the robot’s dynamics. Specifically, the agent’s Euclidean distance to the goal was reduced compared to the model-predictive controller. The2-norm magnitude of the controls was also lower, indicating more efficient control actions. The agent’s performance was quantified by the probability of achieving the goal within a given time horizon. The agent’s ability to learn and adapt to the environment was demonstrated through simulations. The agent’s ability to accurately predict its next observation was crucial for its success. The agent’s performance was evaluated using metrics such as the mean squared error between the predicted and actual observations. The agent’z ability to accurately predict its next observation was crucial for its success. 

The agent’z ability to learn and adapt to the environment was demonstrated through simulations. 

The agent’z ability to accurately predict its next observation was crucial
