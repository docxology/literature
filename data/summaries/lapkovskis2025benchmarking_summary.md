# Benchmarking Dynamic SLO Compliance in Distributed Computing Continuum Systems

**Authors:** Alfreds Lapkovskis, Boris Sedlak, Sindri Magnússon, Schahram Dustdar, Praveen Kumar Donta

**Year:** 2025

**Source:** arxiv

**Venue:** N/A

**DOI:** 10.1109/EDGE67623.2025.00020

**PDF:** [lapkovskis2025benchmarking.pdf](../pdfs/lapkovskis2025benchmarking.pdf)

**Generated:** 2025-12-15 11:14:57

**Validation Status:** ✓ Accepted
**Quality Score:** 1.00


## Classification

- **Category**: Core Theory Math
- **Domain**: Computer Science
- **Confidence**: 0.95
- **Reasoning**: The paper's primary contribution is the benchmarking of dynamic Service Level Objective (SLO) compliance in Distributed Computing Continuum Systems. It focuses on comparing different machine learning algorithms (Active Inference and Reinforcement Learning) based on their theoretical performance and adaptability, rather than developing a specific tool or applying a solution to a particular problem. The core of the paper lies in the theoretical evaluation and comparison of these algorithms, making it a core/theory-based contribution.

---

### OverviewThis summary extracts key information from the paper "Benchmarking Dynamic SLO Compliance in Distributed Computing Continuum Systems" by Lapkovskis et al. (2025). The paper investigates the use of Active Inference (AIF) compared to Reinforcement Learning (RL) algorithms for ensuring Service Level Objectives (SLOs) in Distributed Computing Continuum Systems (DCCS). The research focuses on evaluating AIF’s performance against DQN, A2C, and PPO, simulating dynamic workloads and system changes to assess their adaptability.### MethodologyThe authors introduce a realistic use case: a video conferencing application running on an edge device alongside a WebSocket server streaming videos. The core methodology centers around benchmarking these algorithms against each other, using a set of predefined SLOs. The authors state: "Ensuring SLO compliance in large-scale architectures, such as DCCS, is challenging due to their heterogeneous nature and varying service requirements across different devices and applications." The authors further note: "To improve SLO compliance in DCCS, one possibility is to apply machine learning; however, the design choices are often left to the developer." The authors implement AIF, DQN, A2C, and PPO, and they define the following key parameters: "We consider a realistic DCCS use case: an edge device running a video conferencing application alongside a WebSocket server streaming videos." The authors define the following key metrics: "To quantify SLO compliance, we capture a set of metrics that give insights into the performance and efficiency of the streaming pipeline. These metrics include CPU usage (MCPU), memory usage (Mmem), throughput (Mtp), average latency (Mlat), average render scale factor (Mrs) and thermal state (Mts)." The authors also define the following SLOs: "To ensure high QoS and QoE, the intelligent agent is continuously learning system configurations that comply with the following SLOs. For any metric Mx, where x represents a placeholder for any metric type, variables Mmaxx and Mminx denote the upper and lower SLO thresholds, respectively." The authors use a batch size of32 for each algorithm, and they define the following hyperparameters: "Hyperparameters include a surprise threshold factor of2.0, a weight of past data of0.6, an initial additional surprise of1.0, a graph max indegree of8, an epsilon of1.0, an exploration fraction of0.1, and a learning rate of10−4."### ResultsThe authors found that all four algorithms were able to achieve decent SLO compliance under certain conditions. Specifically, AIF showed remarkable sample efficiency by converging significantly faster than other algorithms. The authors state: "In their turn, PPO and A2C were able to converge to an optimal configuration but required multiples of AIF training time, especially A2C." The authors also observed that DQN suffered from instability and sample inefficiency. The authors state: "Although the initial evaluation cycles showed high performance, it progressively declined over time." The authors also found that AIF’s resource consumption was significantly lower than other algorithms. The authors state: "AIF demonstrates remarkable sample efficiency by converging significantly faster than other algorithms." The authors observed that AIF’s performance was affected by the introduction of dynamic workloads and system changes. The authors state: "To assess the adaptable capabilities of the algorithms, we conduct this experiment where a client is suddenly facing a significantly reduced network bandwidth (1 Mb/s)." The authors found that AIF was able to adapt to the changing conditions, while other algorithms struggled. The authors state: "

In this scenario, PPO and A2C were able to converge to an optimal configuration but required multiples of AIF training time, especially A2C." 

The authors observed that AIF’s performance was affected by the introduction of dynamic workloads and system changes.
