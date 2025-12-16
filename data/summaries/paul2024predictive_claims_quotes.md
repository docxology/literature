# On Predictive planning and counterfactual learning in active inference - Key Claims and Quotes

**Authors:** Aswin Paul, Takuya Isomura, Adeel Razi

**Year:** 2024

**Source:** arxiv

**Venue:** N/A

**DOI:** 10.3390/e26060484

**PDF:** [paul2024predictive.pdf](../pdfs/paul2024predictive.pdf)

**Generated:** 2025-12-15 11:29:38

---

Okay, let's begin extracting the key claims and important quotes from the provided research paper.

## Key Claims and Hypotheses

1.  **The Core Claim:** The paper posits that active inference, a biologically plausible framework, offers a principled approach to understanding intelligent behavior, particularly in decision-making and planning.

2.  **Hybrid Decision-Making Scheme:** The authors propose a mixed model that combines ‘planning’ and ‘learning from experience’ approaches within the active inference framework, aiming to balance computational complexity and data efficiency.

3.  **Addressing Entropic Observations:** The paper highlights the challenge posed by encountering highly ‘entropic’ observations (unexpected observations) in active inference and introduces methods to mitigate this issue.

4.  **Data-Complexity Trade-Off:** The research investigates a data-complexity trade-off between the planning and counterfactual learning schemes, suggesting a balanced approach is optimal.

5.  **Explainable Decision-Making:** The authors aim to develop a framework for decision-making that is explainable, allowing for insights into the underlying mechanisms of intelligent behavior.

## Important Quotes

1.  “Given the rapid advancement of artificial intelligence, understanding the foundations of intelligent behaviour is increasingly important.” (Introduction) - *This quote establishes the context and motivation for the research.*

2.  “Active inference has emerged in neuroscience as a biologically plausible framework Friston [2010], which adopts a different approach to modelling intelligent behaviour compared to other contemporary methods like RL.” (Introduction) - *This quote positions active inference within the broader landscape of intelligent behavior models.*

3.  “Maximising the model evidence becomes challenging when the agent encounters a highly ’entropic’ observation (i.e. an unexpected observation) concerning the agent’s generative (world) model Da Costa et al. [2020], Sajid et al. [2021], Millidge et al. [2020].” (Introduction) - *This quote identifies a key technical challenge in active inference.*

4.  “Furthermore, we introduce a mixed model that navigates the data-complexity trade-off between these strategies, leveraging the strengths of both to facilitate balanced decision-making.” (Introduction) - *This quote summarizes the core contribution of the paper.*

5.  “After a general introduction to generative world models in the next section, we take a closer look at the decision-making aspect of active inference.” (Section 2.3.1) - *This quote indicates the structure of the paper.*

6.  “The action selection in the DPEFE scheme is implemented as follows: After evaluating the expected free energy (EFE, G) of future observations using dynamic programming (cf. Paul et al. [2021]), the agent evaluates the probability distribution for selecting an action u as: P(u|s)DPEFE = σ (−α G(u|s))” (Section 2.3.1) - *This quote details the implementation of the DPEFE scheme.*

7. “The action precision parameter (α) may be tuned to increase/decrease the agent’s action selection confidence.” (Section 2.3.1) - *This quote explains the role of the action precision parameter.*

8. “In the simulations, we observe that the agent learns to navigate the grid within the first ten episodes (Fig.7: A) and quickly solves the grid.” (Section 4.1) - *This quote describes the results of the simulations.*

9. “The agent maintains a higher bias to the DPEFE component (i.e. planning) compared to the CL scheme.” (Section 4.1) - *This quote explains the agent’s bias towards the planning component.*


---

**Note:** This output represents the initial extraction based on the provided text. Further refinement and additional quotes may be necessary after a more thorough review of the full paper. The formatting adheres to the requirements outlined in the prompt.
