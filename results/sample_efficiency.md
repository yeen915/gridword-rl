| Condition | Seeds | Steps to 50% | Steps to 80% | Steps to 90% | Final val. success (last 200k) |
|---|---|---|---|---|---|
| PPO 12x12, BFS shaping | 1 | never | never | never | 34.9% |
| PPO, with moving enemy | 1 | 0.38M | 0.60M | 0.95M | 93.1% |
| PPO, Manhattan shaping | 3 | 0.36M | 0.65M | 0.89M | 96.7% |
| PPO, no visit memory | 3 | 0.40M | never | never | 71.1% |
| PPO, 500 fixed train maps | 2 | 0.34M | 0.69M | never | 83.6% |
| PPO, 50 fixed train maps | 2 | never | never | never | 34.3% |
| PPO (main: BFS-shaped, memory, random maps) | 3 | 0.35M | 0.68M | 0.90M | 93.8% |
| PPO 12x12, sparse | 1 | never | never | never | 22.6% |
| PPO 12x12, sparse + curriculum | 1 | never | never | never | 36.1% |
| PPO, sparse + curriculum | 3 | 0.28M | 0.72M | 0.96M | 94.9% |
| PPO, sparse reward | 3 | 0.37M | 0.71M | 0.88M | 95.6% |
