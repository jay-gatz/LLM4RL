# Audit Report V3

## Counts

- Total scanned: 1982
- Verified: 1007
- Core kept: 125
- Out-of-scope RL-for-LLM: 327
- Borderline manual review: 268
- Removed (unverified/nonacademic/invalid): 754

## Tag Distribution

| Category_Tags | Count |
|---|---:|
| Policy | 110 |
| Planning | 60 |
| Reward | 67 |
| CreditAssignment | 33 |
| WorldModel | 6 |
| Exploration | 43 |
| DatasetGeneration | 28 |
| OfflineRL | 10 |
| ToolUse | 44 |
| Memory | 6 |
| Reasoning | 53 |
| Simulation | 33 |
| TreeSearch | 7 |
| Verifier | 42 |
| Curriculum | 17 |
| LanguageConditioning | 42 |
| MultiAgent | 25 |
| Architecture | 125 |
| HybridRL | 2 |
| AgentTraining | 16 |

## Year Distribution (2022-2026)

| Year | Count |
|---|---:|
| 2026 | 54 |
| 2025 | 39 |
| 2024 | 10 |
| 2023 | 19 |
| 2022 | 3 |

## Coverage Checklist

- Weak clusters (<8 papers): WorldModel, Memory, TreeSearch, HybridRL
- Expansion methods used: PDF seed extraction, targeted arXiv search, OpenAlex citation expansion.
- Gate A papers.csv provenance count: 42
- Gate A pdf_ref provenance count: 37
- Gate B 2022-2024 count: 32
- Gate B 2025 count: 39
- Gate C max duplicate Why text: 1
- Gate C max duplicate Summary text: 1

## Scope Leak Check (core)

- No GRPO/DPO/PPO/RLHF/alignment/post-training leakage detected in core.

## Summary Uniqueness Stats

- Unique Contribution_Summary entries: 125 / 125
- Unique Why_It_Belongs_in_Survey entries: 125 / 125
- Max repetition Contribution_Summary: 1
- Max repetition Why_It_Belongs_in_Survey: 1

## Must-have Classics Present?

- saycan: present (Do As I Can, Not As I Say: Grounding Language in Robotic Affordances)
- code as policies: present (Code as Policies: Language Model Programs for Embodied Control)
- voyager: present (Voyager: An Open-Ended Embodied Agent with Large Language Models)
- eureka: present (Eureka: Human-Level Reward Design via Coding Large Language Models)
- text2reward: present (Text2Reward: Reward Shaping with Language Models for Reinforcement Learning)

## 20 Representative Core Papers

1. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (2022) - https://openalex.org/W4224912544
2. Code as Policies: Language Model Programs for Embodied Control (2023) - https://openalex.org/W4383097638
3. SMART-LLM: Smart Multi-Agent Robot Task Planning using Large Language Models (2023) - https://arxiv.org/abs/2309.10062v2
4. Theory of Mind for Multi-Agent Collaboration via Large Language Models (2023) - https://openalex.org/W4389523767
5. Eureka: Human-Level Reward Design via Coding Large Language Models (2023) - https://arxiv.org/abs/2310.12931v2
6. Guiding Pretraining in Reinforcement Learning with Large Language Models (2023) - https://arxiv.org/abs/2302.06692v2
7. A Survey of Robot Intelligence with Large Language Models (2024) - https://openalex.org/W4403074774
8. Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects (2024) - https://arxiv.org/abs/2401.03428v1
9. On the Prospects of Incorporating Large Language Models (LLMs) in Automated Planning and Scheduling (APS) (2024) - https://openalex.org/W4399177300
10. AutoWebGLM: A Large Language Model-based Web Navigating Agent (2024) - https://arxiv.org/abs/2404.03648v2
11. RL4F: Generating Natural Language Feedback with Reinforcement Learning for Repairing Model Outputs (2023) - https://arxiv.org/abs/2305.08844v2
12. Online Decision Transformer (2022) - https://openalex.org/W4226151089
13. Reward Design with Language Models (2023) - https://openalex.org/W4322825501
14. Skill Reinforcement Learning and Planning for Open-World Long-Horizon Tasks (2023) - https://arxiv.org/abs/2303.16563v2
15. Language Instructed Reinforcement Learning for Human-AI Coordination (2023) - https://arxiv.org/abs/2304.07297v2
16. Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling (2023) - https://arxiv.org/abs/2301.12050v2
17. LLM-Coordination: Evaluating and Analyzing Multi-agent Coordination Abilities in Large Language Models (2023) - https://arxiv.org/abs/2310.03903v3
18. Asynchronous Large Language Model Enhanced Planner for Autonomous Driving (2024) - https://arxiv.org/abs/2406.14556v3
19. Large Language Models as Generalizable Policies for Embodied Tasks (2023) - https://arxiv.org/abs/2310.17722v2
20. Text2Reward: Reward Shaping with Language Models for Reinforcement Learning (2023) - https://openalex.org/W4386977020