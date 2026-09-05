---
license: apache-2.0
task_categories:
- question-answering
tags:
- finance
- insurance
size_categories:
- 10K<n<100K
language:
- en
---
# INS-MMBench Dataset

**INS-MMBench** is the first comprehensive LVLMs benchmark for the insurance domain, proposed in our paper:

> **INS-MMBench: A Comprehensive Benchmark for Evaluating LVLMs' Performance in Insurance**

[📄Paper Link](https://arxiv.org/pdf/2406.09105) | [🐙GitHub Repository](https://github.com/FDU-INS/INS-MMBench)

---

## Overview

INS-MMBench is the first comprehensive LVLMs benchmark for the insurance domain, it covers four representative insurance types: auto, property, health, and agricultural insurance and key insurance stages such as risk underwriting, risk monitoring and claim processing. INS-MMBench consists of three layers task: 
 - Fundamental task, which focuses on the understanding of individual insurance-related visual elements;
 - Meta-task, which involves the compositional understanding of multiple insurance-related visual elements;
 - Scenario task, which pertains to real-world insurance tasks requiring multi-step reasoning and decision-making.

INS-MMBench includes a total of 12,052 images, 10,372 thoroughly designed questions (including multiple-choice visual questions and free-text visual questions), comprehensively covering 5 scenario tasks, 12 meta-tasks and 22 fundamental tasks.

This dataset includes **six TSV files**, each corresponding to a specific task described in our paper:

| File Name | Task Type |
|-----------|-----------|
| `INS_MMBench_fundamental.tsv` | Fundamental Task |
| `multi_step_claim.tsv` | Scenario Task - Auto Insurance Claim Processing |
| `multi_step_liability.tsv` | Scenario Task - Auto Insurance Accident Liability Determination |
| `multi_step_health.tsv` | Scenario Task - Health Insurance Risk Assessment |
| `multi_step_property.tsv` | Scenario Task - Property Insurance Risk Management |
| `multi_step_agri.tsv` | Scenario Task - Agricultural Insurance Claim Processing |

---