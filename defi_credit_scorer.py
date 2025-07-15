#!/usr/bin/env python3
"""
DeFi Wallet Credit Scoring System - LATEST VERSION
==================================================

A machine learning model to assign credit scores (0-1000) to DeFi wallets
based on their transaction behavior on the Aave V2 protocol.

Updated to work with the actual data structure from user-wallet-transactions.json

Author: AI Assistant
Date: July 2025
Version: 2.0 (Updated for actual data structure)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DeFiCreditScorer:
    """
    A comprehensive credit scoring system for DeFi wallets based on transaction behavior.
    Updated to work with the actual data structure.
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = []
        self.score_weights = {
            'consistency': 0.25,
            'responsibility': 0.30,
            'activity': 0.20,
            'risk_management': 0.25
        }
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and parse the JSON transaction data."""
        print("Loading transaction data...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Map fields to standard names
        df['user'] = df['userWallet']  # Map userWallet to user
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['date'] = df['timestamp'].dt.date
        
        # Extract additional data from actionData
        if 'actionData' in df.columns:
            # Parse actionData for amounts and asset information
            action_data_expanded = df['actionData'].apply(lambda x: x if isinstance(x, dict) else {})
            
            # Extract key fields from actionData
            df['amount'] = action_data_expanded.apply(lambda x: x.get('amount', 0))
            df['assetSymbol'] = action_data_expanded.apply(lambda x: x.get('assetSymbol', 'UNKNOWN'))
            df['assetPriceUSD'] = action_data_expanded.apply(lambda x: x.get('assetPriceUSD', 0))
            df['reserve'] = df['assetSymbol']  # Use assetSymbol as reserve
            
            # Convert amount to numeric (handle string amounts)
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
            df['assetPriceUSD'] = pd.to_numeric(df['assetPriceUSD'], errors='coerce').fillna(0)
        else:
            # If no actionData, create default values
            df['amount'] = 1.0
            df['assetSymbol'] = 'UNKNOWN'
            df['assetPriceUSD'] = 0.0
            df['reserve'] = 'UNKNOWN'
        
        print(f"Loaded {len(df)} transactions for {df['user'].nunique()} unique wallets")
        print(f"Networks: {df['network'].unique()}")
        print(f"Actions: {df['action'].value_counts().to_dict()}")
        print(f"Asset symbols: {df['assetSymbol'].value_counts().head().to_dict()}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive features for credit scoring.
        
        Features are grouped into 4 categories:
        1. Consistency Features - Regular usage patterns
        2. Responsibility Features - Repayment behavior 
        3. Activity Features - Transaction volume and frequency
        4. Risk Management Features - Diversification and liquidation history
        """
        print("Engineering features...")
        
        # Group by wallet
        wallet_features = []
        
        for wallet in df['user'].unique():
            wallet_data = df[df['user'] == wallet].copy()
            features = {'wallet': wallet}
            
            # === BASIC STATISTICS ===
            features['total_transactions'] = len(wallet_data)
            features['unique_days_active'] = wallet_data['date'].nunique()
            features['days_since_first_tx'] = (wallet_data['timestamp'].max() - wallet_data['timestamp'].min()).days + 1
            
            # Transaction type distribution
            tx_types = wallet_data['action'].value_counts()
            
            # Handle the actual action types in your data
            action_types = ['deposit', 'borrow', 'repay', 'withdraw', 'liquidation']
            for action in action_types:
                features[f'{action}_count'] = tx_types.get(action, 0)
                features[f'{action}_ratio'] = tx_types.get(action, 0) / len(wallet_data)
            
            # Map some actions to standard names for compatibility
            features['redeemunderlying_count'] = features.get('withdraw_count', 0)
            features['redeemunderlying_ratio'] = features.get('withdraw_ratio', 0)
            features['liquidationcall_count'] = features.get('liquidation_count', 0)
            features['liquidationcall_ratio'] = features.get('liquidation_ratio', 0)
            
            # === CONSISTENCY FEATURES (25%) ===
            # Regular activity patterns
            features['avg_daily_transactions'] = len(wallet_data) / features['days_since_first_tx']
            features['activity_consistency'] = features['unique_days_active'] / features['days_since_first_tx']
            
            # Transaction timing patterns
            wallet_data['hour'] = wallet_data['timestamp'].dt.hour
            features['hour_diversity'] = wallet_data['hour'].nunique() / 24
            features['weekend_activity_ratio'] = len(wallet_data[wallet_data['timestamp'].dt.weekday >= 5]) / len(wallet_data)
            
            # === RESPONSIBILITY FEATURES (30%) ===
            # Repayment behavior
            borrows = wallet_data[wallet_data['action'] == 'borrow']
            repays = wallet_data[wallet_data['action'] == 'repay']
            
            if len(borrows) > 0:
                features['repay_to_borrow_ratio'] = len(repays) / len(borrows)
                
                # Calculate average time to repay (simplified approach)
                if len(repays) > 0:
                    # Use timestamps as integers to avoid datetime issues
                    avg_borrow_timestamp = borrows['timestamp'].astype('int64').mean()
                    avg_repay_timestamp = repays['timestamp'].astype('int64').mean()
                    
                    # Calculate difference in days (nanoseconds to days conversion)
                    time_diff_days = (avg_repay_timestamp - avg_borrow_timestamp) / (1e9 * 86400)
                    features['avg_repay_delay_days'] = max(0, time_diff_days)
                else:
                    features['avg_repay_delay_days'] = 999  # High penalty for no repayments
            else:
                features['repay_to_borrow_ratio'] = 1.0  # No borrows = perfect ratio
                features['avg_repay_delay_days'] = 0
            
            # Liquidation history (negative indicator)
            features['liquidation_ratio'] = features['liquidationcall_count'] / len(wallet_data)
            features['has_liquidations'] = 1 if features['liquidationcall_count'] > 0 else 0
            
            # === ACTIVITY FEATURES (20%) ===
            # Volume and amounts (using actual amount data)
            if 'amount' in wallet_data.columns and wallet_data['amount'].sum() > 0:
                # Convert amounts to USD values where possible
                wallet_data['usd_value'] = wallet_data['amount'] * wallet_data['assetPriceUSD']
                
                features['total_volume_usd'] = wallet_data['usd_value'].sum()
                features['avg_transaction_size_usd'] = wallet_data['usd_value'].mean()
                features['total_volume_tokens'] = wallet_data['amount'].sum()
                features['avg_transaction_size_tokens'] = wallet_data['amount'].mean()
                features['volume_variance'] = wallet_data['usd_value'].var()
                
                # Use log scale for very large amounts
                features['log_total_volume'] = np.log1p(features['total_volume_usd'])
            else:
                # Fallback to transaction counts
                features['total_volume_usd'] = len(wallet_data)
                features['avg_transaction_size_usd'] = 1.0
                features['total_volume_tokens'] = len(wallet_data)
                features['avg_transaction_size_tokens'] = 1.0
                features['volume_variance'] = 0.0
                features['log_total_volume'] = np.log1p(len(wallet_data))
            
            # Transaction frequency patterns
            daily_tx = wallet_data.groupby('date').size()
            features['max_daily_transactions'] = daily_tx.max()
            features['avg_gap_between_transactions'] = features['days_since_first_tx'] / len(wallet_data)
            
            # === RISK MANAGEMENT FEATURES (25%) ===
            # Asset diversification using actual asset symbols
            if 'assetSymbol' in wallet_data.columns:
                unique_assets = wallet_data['assetSymbol'].nunique()
                features['unique_assets'] = unique_assets
                
                # Calculate concentration (higher = more concentrated = riskier)
                if unique_assets > 1:
                    asset_distribution = wallet_data['assetSymbol'].value_counts()
                    features['asset_concentration'] = (asset_distribution.iloc[0] / len(wallet_data))
                else:
                    features['asset_concentration'] = 1.0  # Single asset = full concentration
                
                # Popular assets bonus (USDC, USDT, DAI, WETH, etc.)
                popular_assets = ['USDC', 'USDT', 'DAI', 'WETH', 'WBTC', 'MATIC']
                features['uses_popular_assets'] = wallet_data['assetSymbol'].isin(popular_assets).sum() / len(wallet_data)
            else:
                features['unique_assets'] = 1
                features['asset_concentration'] = 0.5
                features['uses_popular_assets'] = 0.5
            
            # Deposit vs Borrow balance
            deposits = features['deposit_count'] + features.get('withdraw_count', 0)
            borrows_total = features['borrow_count']
            total_financial_actions = deposits + borrows_total + 1  # +1 to avoid division by zero
            features['deposit_to_total_ratio'] = deposits / total_financial_actions
            
            # Risk indicators
            features['high_frequency_user'] = 1 if features['avg_daily_transactions'] > 5 else 0
            features['bot_like_behavior'] = 1 if (features['hour_diversity'] < 0.2 and features['total_transactions'] > 50) else 0
            
            wallet_features.append(features)
        
        feature_df = pd.DataFrame(wallet_features)
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        print(f"Engineered {len(feature_df.columns)-1} features for {len(feature_df)} wallets")
        return feature_df
    
    def calculate_base_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rule-based base scores for each category."""
        df = features_df.copy()
        
        # === CONSISTENCY SCORE (0-250) ===
        consistency_score = (
            # Regular activity bonus
            np.clip(df['activity_consistency'] * 100, 0, 50) +
            # Long-term usage bonus
            np.clip(np.log1p(df['days_since_first_tx']) * 20, 0, 50) +
            # Diverse timing bonus
            df['hour_diversity'] * 50 +
            # Sustained activity bonus
            np.clip(df['unique_days_active'] * 2, 0, 100)
        )
        
        # === RESPONSIBILITY SCORE (0-300) ===
        responsibility_score = (
            # Repayment behavior (most important)
            np.clip(df['repay_to_borrow_ratio'] * 150, 0, 150) +
            # Quick repayment bonus
            np.clip(100 - df['avg_repay_delay_days'], 0, 50) +
            # Liquidation penalty
            np.clip(100 - df['liquidation_ratio'] * 500, 0, 100)
        )
        
        # === ACTIVITY SCORE (0-200) ===
        activity_score = (
            # Transaction volume (using log scale)
            np.clip(df['log_total_volume'] * 10, 0, 100) +
            # Regular usage
            np.clip(df['avg_daily_transactions'] * 20, 0, 50) +
            # Engagement diversity
            np.clip(df['unique_days_active'] * 1, 0, 50)
        )
        
        # === RISK MANAGEMENT SCORE (0-250) ===
        risk_score = (
            # High deposit ratio bonus
            df['deposit_to_total_ratio'] * 100 +
            # Asset diversification bonus
            np.clip(df['unique_assets'] * 20, 0, 50) +
            # Popular assets bonus
            df['uses_popular_assets'] * 50 +
            # Bot behavior penalty
            np.clip(100 - df['bot_like_behavior'] * 100, 0, 50)
        )
        
        # Combine scores
        df['consistency_score'] = np.clip(consistency_score, 0, 250)
        df['responsibility_score'] = np.clip(responsibility_score, 0, 300)
        df['activity_score'] = np.clip(activity_score, 0, 200)
        df['risk_score'] = np.clip(risk_score, 0, 250)
        
        # Calculate final score
        df['credit_score'] = (
            df['consistency_score'] +
            df['responsibility_score'] +
            df['activity_score'] +
            df['risk_score']
        )
        
        # Ensure score is between 0-1000
        df['credit_score'] = np.clip(df['credit_score'], 0, 1000)
        
        return df
    
    def train_model(self, features_df: pd.DataFrame) -> None:
        """Train the ML model on engineered features."""
        print("Training ML model...")
        
        # Get base scores first
        scored_df = self.calculate_base_scores(features_df)
        
        # Prepare features for ML
        feature_cols = [col for col in scored_df.columns 
                       if col not in ['wallet', 'credit_score', 'consistency_score', 
                                    'responsibility_score', 'activity_score', 'risk_score']]
        
        X = scored_df[feature_cols]
        y = scored_df['credit_score']
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model to refine the rule-based scores
        self.model.fit(X_scaled, y)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"Model trained. Top 5 features:")
        print(feature_importance.head())
    
    def predict_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predict credit scores for wallets."""
        # Get base scores
        scored_df = self.calculate_base_scores(features_df)
        
        # Prepare features
        X = scored_df[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Get ML refinement
        ml_scores = self.model.predict(X_scaled)
        
        # Blend rule-based and ML scores
        final_scores = 0.7 * scored_df['credit_score'] + 0.3 * ml_scores
        scored_df['final_credit_score'] = np.clip(final_scores, 0, 1000)
        
        return scored_df
    
    def analyze_scores(self, scored_df: pd.DataFrame) -> Dict:
        """Analyze the distribution and patterns in credit scores."""
        print("Analyzing score distribution...")
        
        scores = scored_df['final_credit_score']
        
        analysis = {
            'total_wallets': len(scored_df),
            'mean_score': scores.mean(),
            'median_score': scores.median(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max()
        }
        
        # Score distribution by ranges
        ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), 
                 (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000)]
        
        range_analysis = {}
        for min_score, max_score in ranges:
            mask = (scores >= min_score) & (scores < max_score)
            count = mask.sum()
            percentage = (count / len(scores)) * 100
            range_analysis[f'{min_score}-{max_score}'] = {
                'count': count,
                'percentage': percentage
            }
        
        analysis['score_ranges'] = range_analysis
        
        # Behavioral analysis
        low_score_wallets = scored_df[scores <= 300]
        high_score_wallets = scored_df[scores >= 700]
        
        analysis['low_score_behavior'] = {
            'avg_liquidation_ratio': low_score_wallets['liquidation_ratio'].mean(),
            'avg_repay_ratio': low_score_wallets['repay_to_borrow_ratio'].mean(),
            'avg_transactions': low_score_wallets['total_transactions'].mean(),
            'bot_like_percentage': (low_score_wallets['bot_like_behavior'].sum() / len(low_score_wallets)) * 100 if len(low_score_wallets) > 0 else 0
        }
        
        analysis['high_score_behavior'] = {
            'avg_liquidation_ratio': high_score_wallets['liquidation_ratio'].mean() if len(high_score_wallets) > 0 else 0,
            'avg_repay_ratio': high_score_wallets['repay_to_borrow_ratio'].mean() if len(high_score_wallets) > 0 else 0,
            'avg_transactions': high_score_wallets['total_transactions'].mean() if len(high_score_wallets) > 0 else 0,
            'avg_consistency': high_score_wallets['activity_consistency'].mean() if len(high_score_wallets) > 0 else 0
        }
        
        return analysis
    
    def create_visualizations(self, scored_df: pd.DataFrame, analysis: Dict) -> None:
        """Create visualization plots for the analysis."""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Score distribution histogram
        axes[0, 0].hist(scored_df['final_credit_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Credit Score Distribution')
        axes[0, 0].set_xlabel('Credit Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(analysis['mean_score'], color='red', linestyle='--', label=f'Mean: {analysis["mean_score"]:.1f}')
        axes[0, 0].legend()
        
        # Score ranges bar chart
        ranges = list(analysis['score_ranges'].keys())
        percentages = [analysis['score_ranges'][r]['percentage'] for r in ranges]
        axes[0, 1].bar(ranges, percentages, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Score Distribution by Ranges')
        axes[0, 1].set_xlabel('Score Range')
        axes[0, 1].set_ylabel('Percentage of Wallets')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Score vs transaction count scatter
        axes[1, 0].scatter(scored_df['total_transactions'], scored_df['final_credit_score'], 
                          alpha=0.6, color='purple')
        axes[1, 0].set_title('Credit Score vs Transaction Count')
        axes[1, 0].set_xlabel('Total Transactions')
        axes[1, 0].set_ylabel('Credit Score')
        
        # Component scores comparison
        components = ['consistency_score', 'responsibility_score', 'activity_score', 'risk_score']
        component_means = [scored_df[comp].mean() for comp in components]
        axes[1, 1].bar(components, component_means, alpha=0.7, color=['orange', 'red', 'blue', 'green'])
        axes[1, 1].set_title('Average Component Scores')
        axes[1, 1].set_ylabel('Average Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('credit_score_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, scored_df: pd.DataFrame, analysis: Dict, output_path: str = 'wallet_scores.json') -> None:
        """Save the results to a JSON file."""
        # Prepare output data
        results = {
            'metadata': {
                'total_wallets': analysis['total_wallets'],
                'scoring_date': datetime.now().isoformat(),
                'score_statistics': {
                    'mean': float(analysis['mean_score']),
                    'median': float(analysis['median_score']),
                    'std': float(analysis['std_score']),
                    'min': float(analysis['min_score']),
                    'max': float(analysis['max_score'])
                }
            },
            'wallet_scores': []
        }
        
        # Add wallet scores
        for _, row in scored_df.iterrows():
            wallet_score = {
                'wallet_address': row['wallet'],
                'credit_score': int(row['final_credit_score']),
                'component_scores': {
                    'consistency': int(row['consistency_score']),
                    'responsibility': int(row['responsibility_score']),
                    'activity': int(row['activity_score']),
                    'risk_management': int(row['risk_score'])
                },
                'key_metrics': {
                    'total_transactions': int(row['total_transactions']),
                    'days_active': int(row['unique_days_active']),
                    'repay_ratio': float(row['repay_to_borrow_ratio']),
                    'liquidation_ratio': float(row['liquidation_ratio'])
                }
            }
            results['wallet_scores'].append(wallet_score)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")

def main():
    """Main execution function."""
    print("=== DeFi Wallet Credit Scoring System ===\n")
    
    # Initialize scorer
    scorer = DeFiCreditScorer()
    
    # Load data
    json_file_path = r"C:\Users\INTEL\OneDrive\Desktop\defi-credit-scoring\user-wallet-transactions.json"
    
    try:
        # Load and process data
        df = scorer.load_data(json_file_path)
        
        # Engineer features
        features_df = scorer.engineer_features(df)
        
        # Train model
        scorer.train_model(features_df)
        
        # Predict scores
        scored_df = scorer.predict_scores(features_df)
        
        # Analyze results
        analysis = scorer.analyze_scores(scored_df)
        
        # Create visualizations
        scorer.create_visualizations(scored_df, analysis)
        
        # Save results
        scorer.save_results(scored_df, analysis)
        
        # Print summary
        print("\n=== SCORING COMPLETE ===")
        print(f"Processed {analysis['total_wallets']} wallets")
        print(f"Average credit score: {analysis['mean_score']:.1f}")
        print(f"Score range: {analysis['min_score']:.1f} - {analysis['max_score']:.1f}")
        
        print("\nScore distribution:")
        for range_name, range_data in analysis['score_ranges'].items():
            print(f"  {range_name}: {range_data['count']} wallets ({range_data['percentage']:.1f}%)")
        
    except FileNotFoundError:
        print(f"Error: Could not find {json_file_path}")
        print("Please download the transaction data file and update the file path.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()