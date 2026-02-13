# Lab 3: Contextual Bandit-Based News Article Recommendation

**Student:** Sandeep Ram  
**Roll Number:** U20230083  
**Branch:** `Sandeep_U20230083`

---

## Overview

This project implements a Contextual Multi-Armed Bandit (CMAB) based news recommendation system. The system classifies users into one of 3 categories acting as the context, then uses bandit algorithms to select the best news category for each user type. The environment has 3 contexts × 4 news categories = 12 arms.

---

## Approach

### Data Preprocessing
- Dropped rows with missing values from all datasets
- Encoded user labels (user_1, user_2, user_3) using LabelEncoder
- Filtered news articles to the 4 required categories: Entertainment, Education, Tech, Crime

### User Classification
- Trained a Random Forest Classifier on 28 numeric features
- 80/20 stratified train/validation split
- Achieved ~89% validation accuracy

### Bandit Algorithms
Implemented three contextual bandit strategies, each maintaining Q-value estimates for all 12 arms:

- **Epsilon-Greedy:** Explores randomly with probability ε, otherwise exploits best arm
- **UCB:** Selects arm with highest upper confidence bound within the context
- **SoftMax:** Selects arms proportional to exp(Q/τ) using Boltzmann distribution

### Recommendation Engine
For each test user: classify → get context → select best category via bandit policy → sample an article from that category.

### Evaluation
- Simulated each algorithm for T=10,000 steps
- Plotted cumulative average reward vs time for all algorithms
- Compared hyperparameters: ε ∈ {0.01, 0.05, 0.1, 0.2} and c ∈ {0.5, 1.0, 2.0, 3.0}

---

## Results

| Algorithm | Avg Reward |
|-----------|-----------|
| Epsilon-Greedy (ε=0.1) | ~1.93 |
| UCB (c=2.0) | ~2.23 |
| SoftMax (τ=1.0) | ~2.09 |

UCB performed best overall. All three algorithms converge to the same optimal category per user context:
- user_1 → Crime, user_2 → Education, user_3 → Tech

Smaller ε values (0.01-0.05) worked best for Epsilon-Greedy. UCB was robust across all tested c values.
