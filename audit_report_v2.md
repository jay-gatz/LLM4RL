# Audit Report V2

## Counts

- Total scanned: 2005
- Verified: 1028
- Core kept: 119
- Out-of-scope RL-for-LLM: 291
- Borderline manual review: 321
- Removed (unverified/nonacademic/invalid): 741

## Tag Distribution

| Category_Tags | Count |
|---|---:|
| Policy | 107 |
| Planning | 58 |
| Reward | 64 |
| CreditAssignment | 33 |
| WorldModel | 6 |
| Exploration | 39 |
| DatasetGeneration | 26 |
| OfflineRL | 9 |
| ToolUse | 42 |
| Memory | 5 |
| Reasoning | 54 |
| Simulation | 31 |
| TreeSearch | 7 |
| Verifier | 43 |
| Curriculum | 15 |
| LanguageConditioning | 37 |
| MultiAgent | 25 |
| Architecture | 119 |
| HybridRL | 1 |
| AgentTraining | 16 |

## Year Distribution (2022-2026)

| Year | Count |
|---|---:|
| 2026 | 55 |
| 2025 | 39 |
| 2024 | 9 |
| 2023 | 15 |
| 2022 | 1 |

## Coverage Checklist

- Weak clusters (<8 papers): WorldModel, Memory, TreeSearch, HybridRL
- Expansion methods used: PDF seed extraction, targeted arXiv search, OpenAlex citation expansion.
- Gate A papers.csv provenance count: 37
- Gate A pdf_ref provenance count: 36
- Gate B 2022-2024 count: 25
- Gate B 2025 count: 39
- Gate C max duplicate Why text: 1
- Gate C max duplicate Summary text: 1

## Scope Leak Check (core)

- No GRPO/DPO/PPO/RLHF/alignment/post-training leakage detected in core.

## Summary Uniqueness Stats

- Unique Contribution_Summary entries: 119 / 119
- Unique Why_It_Belongs_in_Survey entries: 119 / 119
- Max repetition Contribution_Summary: 1
- Max repetition Why_It_Belongs_in_Survey: 1

## Must-have Classics Present?

- saycan: present (Do As I Can, Not As I Say: Grounding Language in Robotic Affordances)
- code as policies: present (Code as Policies: Language Model Programs for Embodied Control)
- voyager: missing in core
- eureka: missing in core
- text2reward: present (Text2Reward: Reward Shaping with Language Models for Reinforcement Learning)

## 20 Representative Core Papers

1. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (2022) - https://openalex.org/W4224912544
2. Code as Policies: Language Model Programs for Embodied Control (2023) - https://openalex.org/W4383097638
3. SMART-LLM: Smart Multi-Agent Robot Task Planning using Large Language Models (2023) - https://arxiv.org/abs/2309.10062v2
4. Theory of Mind for Multi-Agent Collaboration via Large Language Models (2023) - https://openalex.org/W4389523767
5. Guiding Pretraining in Reinforcement Learning with Large Language Models (2023) - https://arxiv.org/abs/2302.06692v2
6. A Survey of Robot Intelligence with Large Language Models (2024) - https://openalex.org/W4403074774
7. Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects (2024) - https://arxiv.org/abs/2401.03428v1
8. On the Prospects of Incorporating Large Language Models (LLMs) in Automated Planning and Scheduling (APS) (2024) - https://openalex.org/W4399177300
9. AutoWebGLM: A Large Language Model-based Web Navigating Agent (2024) - https://arxiv.org/abs/2404.03648v2
10. RL4F: Generating Natural Language Feedback with Reinforcement Learning for Repairing Model Outputs (2023) - https://arxiv.org/abs/2305.08844v2
11. Skill Reinforcement Learning and Planning for Open-World Long-Horizon Tasks (2023) - https://arxiv.org/abs/2303.16563v2
12. From Decision to Action in Surgical Autonomy: Multi-Modal Large Language Models for Robot-Assisted Blood Suction (2025) - https://openalex.org/W4406022307
13. Language Instructed Reinforcement Learning for Human-AI Coordination (2023) - https://arxiv.org/abs/2304.07297v2
14. Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling (2023) - https://arxiv.org/abs/2301.12050v2
15. LLM-Coordination: Evaluating and Analyzing Multi-agent Coordination Abilities in Large Language Models (2023) - https://arxiv.org/abs/2310.03903v3
16. Asynchronous Large Language Model Enhanced Planner for Autonomous Driving (2024) - https://arxiv.org/abs/2406.14556v3
17. Large Language Models as Generalizable Policies for Embodied Tasks (2023) - https://arxiv.org/abs/2310.17722v2
18. Text2Reward: Reward Shaping with Language Models for Reinforcement Learning (2023) - https://openalex.org/W4386977020
19. Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks (2024) - https://arxiv.org/abs/2405.01534v1
20. Unleashing the Power of Pre-trained Language Models for Offline Reinforcement Learning (2023) - https://arxiv.org/abs/2310.20587v5