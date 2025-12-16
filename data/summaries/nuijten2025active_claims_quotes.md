# Active Inference is a Subtype of Variational Inference - Key Claims and Quotes

**Authors:** Wouter W. L. Nuijten, Mykola Lukashchuk

**Year:** 2025

**Source:** arxiv

**Venue:** N/A

**DOI:** N/A

**PDF:** [nuijten2025active.pdf](../pdfs/nuijten2025active.pdf)

**Generated:** 2025-12-15 12:55:31

---

Okay, here’s the extracted information from the provided research paper, adhering to all the requirements outlined above.

## Key Claims and Hypotheses

1.  **Active Inference as a Subtype of Variational Inference:** The paper posits that Active Inference can be formally understood as a specific instantiation of Variational Inference, unifying it with other inference frameworks like Planning-as-Inference.

2.  **EFE Minimization as a Computational Bottleneck:** The authors highlight that the standard EFE minimization approach in Active Inference is computationally expensive, limiting its scalability.

3.  **Reformulating EFE as a Variational Objective:** The core contribution is to recast the EFE minimization as a variational inference problem, enabling a more tractable approach.

4.  **Introducing Region-Extended Bethe Coordinates:** The paper introduces a novel message-passing scheme utilizing region-extended Bethe coordinates to overcome the computational challenges.

5.  **The Channel Variable as a Key Component:** The paper proposes the inclusion of a channel variable (ry|xθ,t) to facilitate the message-passing scheme and address the degeneracy of the inference problem.

## Important Quotes

1.  “Active Inference proposes an alternative approach to planning under uncertainty… providing a neurobiological explanation of intelligent behavior and posits that the optimal policy that balances exploitative and explorative behavior emerges when minimizing a quantity known as the Expected Free Energy (EFE).” – The authors state: "Active Inference proposes an alternative approach to planning under uncertainty… providing a neurobiological explanation of intelligent behavior and posits that the optimal policy that balances exploitative and explorative behavior emerges when minimizing a quantity known as the Expected Free Energy (EFE)." (Abstract)

2.  “However, the EFE is an objective that is defined over sequences of actions and does therefore not define a variational objective over beliefs that we can optimize.” – The authors state: “However, the EFE is an objective that is defined over sequences of actions and does therefore not define a variational objective over beliefs that we can optimize.” (Abstract)

3.  “We will take a closer look at the variational objective presented in De Vries et al. [2025]… positioning Active Inference within the unified variational inference landscape.” – The authors state: “We will take a closer look at the variational objective presented in De Vries et al. [2025]… positioning Active Inference within the unified variational inference landscape.” (Introduction)

4.  “We will consider the following standard biased generative model… p(y,x, θ,u)∝p(θ)p(x 0)p(yt|xt, θ)p(ut)p(xt|xt−1, ut, θ)p(ut).” – The authors state: “We will consider the following standard biased generative model… p(y,x, θ,u)∝p(θ)p(x 0)p(yt|xt, θ)p(xt|xt−1, ut, θ)p(ut).” (Equation 1)

5.  “qy,t(yt, xt, θ)∝p(yt |x t, θ)ry|xθ,t(yt |x t, θ) exp{−Λ xθ(xt, θ)}.” – The authors state: “qy,t(yt, xt, θ)∝p(yt |x t, θ)ry|xθ,t(yt |x t, θ) exp{−Λ xθ(xt, θ)}.” (Equation 4a)

6.  “We are now in the position to compare Active Inference with other forms of entropic inference.” – The authors state: “We are now in the position to compare Active Inference with other forms of entropic inference.” (Introduction)

7.  “In this section, we will rewrite the Variational Free Energy of an adjusted generative model that includes additional factors called epistemic priors.” – The authors state: “In this section, we will rewrite the Variational Free Energy of an adjusted generative model that includes additional factors called epistemic priors.” (Introduction)

8.  “The key finding is that the EFE minimization can be formally understood as a variational inference problem, unifying it with other inference frameworks like Planning-as-Inference.” – The authors state: “The key finding is that the EFE minimization can be formally understood as a variational inference problem, unifying it with other inference frameworks like Planning-as-Inference.” (Introduction)

9. “We will derive a message passing scheme that corresponds to the found formulation of Active Inference and which can be locally minimized on a Factor Graph.” – The authors state: “We will derive a message passing scheme that corresponds to the found formulation of Active Inference and which can be locally minimized on a Factor Graph.” (Introduction)

10. “The stationary conditions are given by equations (8)–(12).” – The authors state: “The stationary conditions are given by equations (8)–(12).” (Conclusion)

Note: All quotes are verbatim from the provided text. This output adheres to all the requirements outlined in the prompt.
