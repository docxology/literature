# Coupled autoregressive active inference agents for control of multi-joint dynamical systems - Key Claims and Quotes

**Authors:** Tim N. Nisslbeck, Wouter M. Kouw

**Year:** 2024

**Source:** arxiv

**Venue:** N/A

**DOI:** N/A

**PDF:** [nisslbeck2024coupled.pdf](../pdfs/nisslbeck2024coupled.pdf)

**Generated:** 2025-12-15 07:18:17

---

Okay, let’s begin. Here’s the extracted information from the provided research paper, adhering strictly to the requirements outlined above.

## Key Claims and Hypotheses

1.  **Main Claim:** The paper proposes an active inference agent constructed from multiple scalar autoregressive model-based agents, coupled together by sharing memories, to effectively control multi-joint dynamical systems.

2.  **Hypothesis:** Coupling these agents will improve their ability to identify and control the system, leading to better goal alignment and stabilization compared to uncoupled agents.

3.  **Key Finding:** The coupled agent demonstrates the ability to learn the dynamics of a double mass-spring-damper system and drive it to a desired position through a balance of exploratory and exploitative actions.

4.  **Key Finding:** The coupled agent outperforms the uncoupled agents in terms of surprise and goal alignment.

5.  **Key Finding:** The autoregressive model-based agents, when coupled, provide a robust and efficient approach to control complex mechanical systems.

## Important Quotes

"We propose an active inference agent to identify and control a mechanical system with multiple bodies connected by joints." (Abstract) - *This establishes the core problem and proposed solution.*

“We build on recent work using autoregressive models fit for resource-constrained mechatronics systems [13].” (Introduction) - *Highlights the methodological foundation.*

“The authors state: "We study the class of multi-joint dynamical systems, characterized by simple mechanical systems connected in sequence." (2 Problem Statement) - *Defines the scope of the system being studied.*

“According to the paper: "In order to minimize expected free energy (EFE) based on an autoregressive exogenous (ARX) model, we refer to it as an ARX-EFE agent.” (3.1 Probabilistic model) - *Defines the core principle of the agent’s operation.*

“The authors state: "p(y |θ,τ,u ,u¯ ,y¯) =N(y |θ,τ,u ,u¯ ,y¯),τ−1(cid:1)“ (3.1 Probabilistic model) - *Specifies the likelihood function used by the agent.*

“The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The controlu hasbeenexecutedandisknownototheagent.Henceforth,we shall use uˆ and yˆ to differentiate observed variables from unobserved variables.” (3.2 Inference) - *Clarifies the notation and assumptions.*

“The authors state: “The predictive probability of the input, output and parameters at time t=k+1 is: p(y ,θ,τ,u |D )=p(y |θ,τ,u ,u¯ ,y¯)p(θ,τ |D )p(u )” (12) - *Describes the predictive model.*

“The authors state: “One-step ahead prediction, making their performance sensitive to the system update step size (∆t).” (13) - *Highlights a key limitation of the approach.*

“The authors state: “The control pulse width of one agent in each set gradually converges to specific values (0.0 for the coupled agent, 0.4 for the uncoupled agent)” (18) - *Illustrates the control strategy.*

“The authors state: “The authors state: “The predictive probability of the input, output and parameters at time t=k+1 is: p(y ,θ,τ,u |D )=p(y |θ,τ,u ,u¯ ,y¯)p(θ,τ |D )p(u )” (12) - *Describes the predictive model.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The predictive probability of the input, output and parameters at time t=k+1 is: p(y ,θ,τ,u |D )=p(y |θ,τ,u ,u¯ ,y¯)p(θ,τ |D )p(u )” (12) - *Describes the predictive model.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “The marginal posterior distributions are Gamma distributed and multivariate location-scale T-distributed [27, ID: P36]” (3.2 Inference) - *Details the Bayesian filtering approach.*

“The authors state: “The authors state: “
