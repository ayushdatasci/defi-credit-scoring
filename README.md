# DeFi Wallet Credit Scoring System

A machine learning-based credit scoring system for DeFi wallets using Aave V2 transaction data.

## Overview

This system analyzes DeFi transaction patterns to assign credit scores (0-1000) to wallet addresses, helping identify reliable users versus risky or bot-like behavior.

## Architecture

### 1. Data Processing Pipeline
```
Raw JSON Data → Feature Engineering → Rule-Based Scoring → ML Refinement → Final Scores
```

### 2. Scoring Components

The credit score is calculated using four weighted components:

#### **Consistency Score (25%)**
- **Activity Consistency**: Regular usage patterns over time
- **Temporal Diversity**: Transaction timing across different hours/days
- **Long-term Engagement**: Sustained activity over extended periods

#### **Responsibility Score (30%)**
- **Repayment Behavior**: Ratio of repayments to borrows
- **Repayment Timing**: Speed of loan repayments
- **Liquidation History**: Penalty for liquidation events

#### **Activity Score (20%)**
- **Transaction Volume**: Total number of transactions
- **Engagement Frequency**: Average daily transaction rate
- **Platform Usage**: Unique days active on the protocol

#### **Risk Management Score (25%)**
- **Asset Diversification**: Number of different assets used
- **Deposit Ratio**: Preference for deposits vs borrows
- **Bot Detection**: Identification of automated/bot-like behavior

### 3. Model Architecture

**Hybrid Approach**: Combines rule-based scoring with machine learning refinement
- **Rule-based Component (70%)**: Transparent, interpretable business logic
- **ML Component (30%)**: Random Forest model for pattern recognition and score refinement

## Features Engineered

### Transaction Behavior Features
- Transaction type distribution (deposit, borrow, repay, etc.)
- Transaction frequency and timing patterns
- Volume and amount statistics
- Asset usage diversity

### Temporal Features
- Days since first transaction
- Activity consistency over time
- Weekend vs weekday activity patterns
- Hour-of-day usage diversity

### Risk Indicators
- Liquidation event frequency
- Borrow-to-repay ratios
- High-frequency trading patterns
- Bot-like behavior detection

### Relationship Features
- Cross-action correlations
- Sequential transaction patterns
- Time-based transaction clustering

## Score Interpretation

| Score Range | Interpretation | Characteristics |
|-------------|----------------|-----------------|
| 800-1000 | Excellent | Consistent, responsible users with diverse activity |
| 600-799 | Good | Regular users with occasional risk factors |
| 400-599 | Average | Mixed behavior with some concerning patterns |
| 200-399 | Poor | High risk indicators, irregular patterns |
| 0-199 | Very Poor | Bot-like behavior, liquidations, or exploitative usage |

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Scorer
```bash
# Download the transaction data file first
python defi_credit_scorer.py
```

### Input Format
Expects JSON file with transaction records containing:
- `user`: Wallet address
- `action`: Transaction type (deposit, borrow, repay, etc.)
- `timestamp`: Unix timestamp
- `reserve`: Asset identifier (optional)
- `amount`: Transaction amount (optional)

### Output Format
Generates `wallet_scores.json` with:
```json
{
  "metadata": {
    "total_wallets": 1000,
    "score_statistics": {...}
  },
  "wallet_scores": [
    {
      "wallet_address": "0x123...",
      "credit_score": 750,
      "component_scores": {
        "consistency": 180,
        "responsibility": 250,
        "activity": 150,
        "risk_management": 170
      },
      "key_metrics": {...}
    }
  ]
}
```

## Model Validation

### Cross-Validation Strategy
- **Temporal Validation**: Split data by time periods to simulate real-world deployment
- **Feature Importance Analysis**: Random Forest feature importance rankings
- **Score Distribution Analysis**: Ensure reasonable score distributions across ranges

### Performance Metrics
- **Score Stability**: Consistency of scores across similar user profiles
- **Business Logic Validation**: Manual review of extreme scores
- **Distribution Analysis**: Score spread across expected ranges

## Key Design Decisions

### 1. Hybrid Scoring Approach
**Why**: Combines interpretability of rule-based systems with pattern recognition of ML
- Rule-based component ensures business logic transparency
- ML component captures complex, non-linear patterns

### 2. Component-Based Architecture
**Why**: Modular design allows for easy debugging and feature importance analysis
- Each component targets specific behavioral aspects
- Weighted combination allows for business priority adjustments

### 3. Robust Feature Engineering
**Why**: DeFi data can be noisy and sparse
- Multiple proxy metrics for reliability
- Temporal smoothing for activity metrics
- Outlier-resistant scaling methods

### 4. Conservative Scoring Philosophy
**Why**: Credit scoring requires conservative risk assessment
- Penalize risky behaviors more heavily than rewarding good behaviors
- Unknown patterns default to neutral scores
- Multiple validation layers prevent gaming

## Extensibility

### Adding New Features
1. Extend `engineer_features()` method
2. Update component score calculations
3. Retrain model with new feature set

### Adjusting Score Weights
Modify `score_weights` dictionary in the class constructor:
```python
self.score_weights = {
    'consistency': 0.25,    # Adjust these values
    'responsibility': 0.30,
    'activity': 0.20,
    'risk_management': 0.25
}
```

### Supporting New Protocols
1. Add protocol-specific action types
2. Implement protocol-specific feature engineering
3. Adjust risk management calculations

## Limitations & Future Improvements

### Current Limitations
- Limited to transaction-level data (no amount/USD values)
- No cross-protocol activity analysis
- Simple temporal pattern detection

### Future Enhancements
- **Multi-Protocol Scoring**: Aggregate behavior across multiple DeFi protocols
- **Real-Time Updates**: Streaming score updates as new transactions occur
- **Ensemble Methods**: Combine multiple ML models for improved accuracy
- **External Data Integration**: Incorporate market conditions and protocol health metrics

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with detailed description

## Contact

For questions or support, please open an issue in the GitHub repository.
