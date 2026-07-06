# Backtesting notes for future version

The project should not be evaluated like a normal price-prediction model only.

Key evaluation metrics:

- Crisis recall: how many mini-crisis periods were detected?
- False alarm rate: how often did the system call stress without a future drawdown?
- Lead time: how many days before the drawdown did the alert appear?
- Drawdown avoided in paper simulation.
- Regime stability: does the model flip too often?

Future paper-simulation rules:

```text
Risk-On: simulated exposure high
Neutral: simulated exposure low/moderate
Risk-Off: simulated exposure defensive
Mini-Crisis Watch: simulated cash/stable allocation
Crisis: observe only
Recovery: gradual simulated re-entry
```
