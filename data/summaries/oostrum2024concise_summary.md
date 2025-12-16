# A Concise Mathematical Description of Active Inference in Discrete Time

**Authors:** Jesse van Oostrum, Carlotta Langer, Nihat Ay

**Year:** 2024

**Source:** arxiv

**Venue:** N/A

**DOI:** 10.1016/j.jmp.2025.102921

**PDF:** [oostrum2024concise.pdf](../pdfs/oostrum2024concise.pdf)

**Generated:** 2025-12-15 11:24:44

**Validation Status:** ✓ Accepted
**Quality Score:** 0.80


## Classification

- **Category**: Core Theory Math
- **Domain**: active inference
- **Confidence**: 0.95
- **Reasoning**: The paper presents a ‘concise mathematical description’ of active inference, focusing on the core theoretical aspects – the generative model, the action selection mechanism, and the free energy function. The emphasis on precise notation and the inclusion of a worked example strongly indicate a contribution to the theoretical foundations of active inference, rather than a practical implementation or application.

---

Okay, here's a summary of the paper "A Concise Mathematical Description of Active Inference in Discrete Time."**Overview**This paper presents a concise mathematical description of active inference in discrete time. The main part of the paper serves as a basic introduction to the topic, including a detailed example of the action selection mechanism. The appendix provides more subtle mathematical details, targeting readers who have already studied the active inference literature. The paper emphasizes precise and standard mathematical notation, ensuring consistency with existing texts and linking all equations to widely used references on active inference.**Key Concepts and Arguments**The authors define active inference as the process of an agent selecting actions and learning models to achieve its goals. They introduce the core mathematical framework for this process, focusing on the following key elements:***Beliefs and Actions:** The agent’s actions are determined by a policy πt, which is a mapping from the current state to an action.***Expected Free Energy:** The core of the model is the expected free energy G(πt|o1:t, a1:t−1), which represents the agent’s subjective assessment of its actions. This is defined as the expected difference between the agent’s predictions and the actual observations.***KL Divergence:** The authors use KL divergence to quantify the difference between the agent’s belief about the world and the preferred world model.***Bayes’ Rule:** The paper relies heavily on Bayes’ rule to update beliefs about the environment and the agent’s own actions.**Specifically, the paper describes:***The agent’s belief about the environment, represented by the probability distribution q(s1:T |o1:t, a1:t−1).*The agent’s action selection policy, πt.*The calculation of the expected free energy, G(πt|o1:t, a1:t−1).*The use of KL divergence to regularize the model.**The paper’s key contribution is to provide a clear and concise mathematical formulation of active inference, making it accessible to a wider audience.****In essence, the paper provides a framework for understanding how an agent can learn and act in an environment by minimizing its subjective “free energy.”****Note:** The paper includes a worked example to illustrate the concepts.---This summary captures the essence of the paper, highlighting its key concepts, arguments, and contributions. It adheres to the instructions provided, focusing on extracting and presenting the core information from the text.
