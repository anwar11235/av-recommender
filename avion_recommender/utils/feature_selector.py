from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import shap

class FeatureSelector:
    """Feature selection and importance analysis for the recommender system."""
    
    def __init__(self):
        self.feature_importance_scores = {}
        self.selected_features = {}
        self.shap_values = None
        
    def analyze_feature_importance(self, 
                                 X: pd.DataFrame, 
                                 y: pd.Series,
                                 method: str = 'random_forest') -> Dict[str, float]:
        """
        Analyze feature importance using specified method.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            method: Method to use ('random_forest', 'mutual_info', or 'shap')
            
        Returns:
            Dictionary of feature importance scores
        """
        if method == 'random_forest':
            return self._random_forest_importance(X, y)
        elif method == 'mutual_info':
            return self._mutual_info_importance(X, y)
        elif method == 'shap':
            return self._shap_importance(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def select_features(self,
                       X: pd.DataFrame,
                       y: pd.Series,
                       n_features: int = 20,
                       method: str = 'random_forest') -> List[str]:
        """
        Select top n features using specified method.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Number of features to select
            method: Method to use for selection
            
        Returns:
            List of selected feature names
        """
        importance_scores = self.analyze_feature_importance(X, y, method)
        sorted_features = sorted(importance_scores.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        
        selected = [f[0] for f in sorted_features[:n_features]]
        self.selected_features[method] = selected
        return selected
    
    def get_feature_importance_report(self,
                                    X: pd.DataFrame,
                                    y: pd.Series) -> pd.DataFrame:
        """
        Generate comprehensive feature importance report using multiple methods.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            
        Returns:
            DataFrame with feature importance scores from different methods
        """
        methods = ['random_forest', 'mutual_info', 'shap']
        importance_dict = {}
        
        for method in methods:
            scores = self.analyze_feature_importance(X, y, method)
            importance_dict[f'{method}_importance'] = scores
            
        report = pd.DataFrame(importance_dict)
        
        # Add aggregate score
        report['aggregate_importance'] = report.mean(axis=1)
        report = report.sort_values('aggregate_importance', ascending=False)
        
        return report
    
    def plot_feature_importance(self,
                              X: pd.DataFrame,
                              y: pd.Series,
                              method: str = 'shap',
                              top_n: int = 20):
        """
        Plot feature importance visualization.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            method: Method to use for importance calculation
            top_n: Number of top features to show
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if method == 'shap':
            self._plot_shap_importance(X, top_n)
        else:
            importance_scores = self.analyze_feature_importance(X, y, method)
            sorted_scores = dict(sorted(importance_scores.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:top_n])
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x=list(sorted_scores.values()), 
                       y=list(sorted_scores.keys()))
            plt.title(f'Top {top_n} Feature Importance ({method})')
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()
    
    def _random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate feature importance using Random Forest."""
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance_scores = dict(zip(X.columns, rf.feature_importances_))
        self.feature_importance_scores['random_forest'] = importance_scores
        return importance_scores
    
    def _mutual_info_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate feature importance using Mutual Information."""
        mi_scores = mutual_info_regression(X, y)
        importance_scores = dict(zip(X.columns, mi_scores))
        
        self.feature_importance_scores['mutual_info'] = importance_scores
        return importance_scores
    
    def _shap_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate feature importance using SHAP values."""
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        explainer = shap.TreeExplainer(rf)
        self.shap_values = explainer.shap_values(X)
        
        importance_scores = dict(zip(X.columns, 
                                   np.abs(self.shap_values).mean(axis=0)))
        self.feature_importance_scores['shap'] = importance_scores
        return importance_scores
    
    def _plot_shap_importance(self, X: pd.DataFrame, top_n: int):
        """Plot SHAP feature importance."""
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Run analyze_feature_importance with method='shap' first.")
        
        shap.summary_plot(self.shap_values, X, max_display=top_n, show=False)
        plt.title(f'Top {top_n} Feature Importance (SHAP)')
        plt.tight_layout()
        plt.show() 