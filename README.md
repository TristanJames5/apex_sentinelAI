# Apex Sentinel

Apex Sentinel is a Tier-1 Hybrid AI Trading Engine that combines Supervised Machine Learning (XGBoost) and Deep Reinforcement Learning (PPO).

### Architecture
1. **The Analyst (XGBoost)**: Calculates 50+ technical indicators and outputs a Win Probability for a Long/Short setup.
2. **The Fund Manager (PPO)**: Receives the Analyst's probability score alongside current market volatility and open PnL. Makes the final risk-management decision on position sizing (1%, 5%, 10%) or closing trades early to protect capital.

This allows the system to scale its risk dynamically and avoid "exploding gradient" math errors commonly found in raw price-based RL engines.
