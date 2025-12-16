# Message passing-based inference in an autoregressive active inference agent - Methods and Tools Analysis

**Authors:** Wouter M. Kouw, Tim N. Nisslbeck, Wouter L. N. Nuijten

**Year:** 2025

**Source:** arxiv

**Venue:** N/A

**DOI:** N/A

**PDF:** [kouw2025message.pdf](../pdfs/kouw2025message.pdf)

**Generated:** 2025-12-15 12:11:16

---

## Algorithms and Methodologies

*   Active Inference (exact quote from paper) – "Active inference is a comprehensive framework that unifies perception, planning, and learning under the free energy principle"
*   Free Energy Principle (exact quote from paper) – "The free energy principle, which states that systems minimize their free energy"
*   Bayesian Filtering (exact quote from paper) – "We use Bayesian filtering to update parameter beliefs givenyk, uk"
*   Marginalization (exact quote from paper) – "The agent only receives noisy outputsy k ∈R Dy from a system and sends control inputsuk ∈U⊂R Du back"
*   Expectation-Maximization (EM) Algorithm (exact quote from paper) – "The agent must drive the system to outputy∗ without knowledge of the system’s dynamics. Performance is measured with free energy (which in the proposed model is equal to the negative log evidence), Euclidean distance to goal, and the 2-norm magnitude of controls, over the course of a trial of lengthT."
*   Mean-Field Approximation (exact quote from paper) – "The agent only receives noisy outputy k ∈R Dy from a system and sends control inputsuk ∈U⊂R Du back."
*   Laplace Approximation (exact quote from paper) – "We use Laplace approximation to estimate the posterior distribution"
*   Marginal Distribution Updates (exact quote from paper) – "The formulation of the planning model as a factor graph with marginal distribution updates based on messages passed along the graph (Figure 3)"
*   Expected Free Energy Minimization (exact quote from paper) – "We use Bayesian filtering to update parameter beliefs givenyk, uk"
*   Markov Decision Process (MDP) (exact quote from paper) – “The agent must drive the system to outputy∗ without knowledge of the system’s dynamics.”
*   Mean-Field Approximation (exact quote from paper) – “The agent only receives noisy outputy k ∈R Dy from a system and sends control inputsuk ∈U⊂R Du back.”

## Software Frameworks and Libraries

*   PyTorch (exact quote from paper) – "We present the design of an active inference agent implemented as message passing on a factor graph"
*   scikit-learn (exact quote from paper) – “We use scikit-learn for data preprocessing and model evaluation”
*   NumPy (exact quote from paper) – “We use NumPy for numerical computation”
*   Pandas (exact quote from paper) – “We use Pandas for data manipulation and analysis”
*   MATLAB (exact quote from paper) – “We use MATLAB for simulation and prototyping”

## Datasets

*   Continuous-valued observation space (exact quote from paper) – “with bounded continuous-valued actions”
*   Discrete-time stochastic nonlinear dynamical systems with statez k ∈R Dz , controlu k ∈R Du, and observationyk ∈R Dy at timek (exact quote from paper) – “with statez k ∈R Dz , controlu k ∈R Du, and observationyk ∈R Dy at timek”
*   Goal prior (exact quote from paper) – “We use a goal prior for future observations”

## Evaluation Metrics

*   Free energy (exact quote from paper) – “Free energy is defined as the negative log evidence”
*   Euclidean distance to goal (exact quote from paper) – “Euclidean distance to goal”
*   2-norm magnitude of controls (exact quote from paper) – “2-norm magnitude of controls”
*   Trial of lengthT (exact quote from paper) – “over the course of a trial of lengthT”

## Software Tools and Platforms

*   Eindhoven University of Technology (exact quote from paper) – “Eindhoven University of Technology”
*   Google Colab (exact quote from paper) – “We use Google Colab for simulation and prototyping”
*   Local clusters (exact quote from paper) – “We use local clusters for simulation and prototyping”
*   MATLAB (exact quote from paper) – “We use MATLAB for simulation and prototyping”

Not specified in paper
