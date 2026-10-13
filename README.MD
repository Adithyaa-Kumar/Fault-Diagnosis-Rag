# Research Charter: Fault-Layer Diagnosis in Parallel Fan-In RAG

## Working Title
Evaluating Dependence-Aware Counterfactual Interventions for Fault-Layer Diagnosis in Parallel Fan-In Multi-Agent RAG

## Core Objective
Determine whether evidence redundancy ($C$) and provenance dependence ($P$) make fault-layer diagnosis less identifiable in parallel fan-in multi-agent RAG, and test whether cluster-level counterfactual interventions improve diagnosis over item-level attribution at an acceptable computational cost.

## Target Architecture
Parallel Fan-In Multi-Agent RAG:
- 1 Query Routing / Retrieval Layer
- 3 Independent Specialist Agents (processing split context windows)
- 1 Arbiter Agent (synthesizing specialist outputs into final verdict)

## Experimental Variables
- **Independent Variables:** 
  - Content Redundancy ($C$: Low vs. High)
  - Provenance Dependence ($P$: Low vs. High)
- **Fault Injections ($F$):**
  - Retrieval Fault ($F_{ret}$)
  - Specialist Reasoning Fault ($F_{spec}$)
  - Arbiter Policy Fault ($F_{arb}$)
- **Dependent Variables:** 
  - Total Trajectories
  - Failed Trajectories
  - LOO-Ambiguity Rate
  - Cluster-Ablation Repair Rate
  - Fault-Layer Diagnosis Accuracy (%)
  - Token & API Call Cost