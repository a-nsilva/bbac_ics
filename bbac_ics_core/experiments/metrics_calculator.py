#!/usr/bin/env python3
"""
BBAC ICS Framework - Metrics Calculator
Centralized metrics computation for experiments.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from typing import List, Tuple

from ..utils.data_structures import (
    ClassificationMetrics,
    LatencyMetrics,
    PerformanceMetrics,
    StatisticalTest
)


class MetricsCalculator:
    """Calculate classification, latency, and performance metrics."""
    
    def calculate_classification_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_scores: List[float] = None
    ) -> ClassificationMetrics:
        """
        Calculate classification metrics.
        
        Args:
            y_true: Ground truth labels (binary: 0=deny, 1=allow)
            y_pred: Predicted labels
            y_scores: Prediction scores (optional, for ROC/PR curves)
            
        Returns:
            ClassificationMetrics object
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # ROC/PR curves (if scores provided)
        roc_auc = 0.0
        avg_precision = 0.0
        fpr_list = []
        tpr_list = []
        precision_curve = []
        recall_curve = []
        
        if y_scores is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_scores)
                avg_precision = average_precision_score(y_true, y_scores)
                
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                fpr_list = fpr.tolist()
                tpr_list = tpr.tolist()
                
                precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
                precision_curve = precision_vals.tolist()
                recall_curve = recall_vals.tolist()
            except ValueError:
                pass  # Handle case where only one class present
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            avg_precision=avg_precision,
            tp=int(tp),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            fpr=fpr_list,
            tpr=tpr_list,
            precision_curve=precision_curve,
            recall_curve=recall_curve
        )
    
    def calculate_latency_metrics(
        self,
        latencies: List[float]
    ) -> LatencyMetrics:
        """
        Calculate latency statistics.
        
        Args:
            latencies: List of latency values (milliseconds)
            
        Returns:
            LatencyMetrics object
        """
        latencies_arr = np.array(latencies)
        
        return LatencyMetrics(
            mean=float(np.mean(latencies_arr)),
            std=float(np.std(latencies_arr)),
            p50=float(np.percentile(latencies_arr, 50)),
            p95=float(np.percentile(latencies_arr, 95)),
            p99=float(np.percentile(latencies_arr, 99)),
            values=latencies
        )
    
    def calculate_performance_metrics(
        self,
        latencies: List[float],
        total_requests: int,
        total_time: float
    ) -> PerformanceMetrics:
        """
        Calculate overall performance metrics.
        
        Args:
            latencies: List of latencies
            total_requests: Total number of requests processed
            total_time: Total execution time (seconds)
            
        Returns:
            PerformanceMetrics object
        """
        latency_metrics = self.calculate_latency_metrics(latencies)
        throughput = total_requests / total_time if total_time > 0 else 0.0
        
        return PerformanceMetrics(
            latency=latency_metrics,
            throughput=throughput,
            total_requests=total_requests,
            total_time=total_time
        )


    def wilcoxon_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        alpha: float = 0.05
    ) -> StatisticalTest:
        """
        Perform Wilcoxon signed-rank test (paired samples).
        
        Tests if two methods produce significantly different scores
        on the same dataset.
        
        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B
            alpha: Significance level
            
        Returns:
            StatisticalTest object
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Scores must have same length for paired test")
        
        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(scores_a, scores_b)
        
        # Effect size (rank-biserial correlation)
        n = len(scores_a)
        r = 1 - (2 * statistic) / (n * (n + 1))
        
        # Confidence interval (approximation)
        z_critical = stats.norm.ppf(1 - alpha / 2)
        se = np.sqrt(n * (n + 1) * (2 * n + 1) / 6)
        
        ci_low = r - z_critical * (1 / se)
        ci_high = r + z_critical * (1 / se)
        
        significant = p_value < alpha
        
        return StatisticalTest(
            p_value=float(p_value),
            effect_size=float(r),
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            significant=significant
        )
    
    def mann_whitney_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        alpha: float = 0.05
    ) -> StatisticalTest:
        """
        Perform Mann-Whitney U test (independent samples).
        
        Tests if two independent methods have different distributions.
        
        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B
            alpha: Significance level
            
        Returns:
            StatisticalTest object
        """
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            scores_a,
            scores_b,
            alternative='two-sided'
        )
        
        # Effect size (rank-biserial correlation)
        n1 = len(scores_a)
        n2 = len(scores_b)
        r = 1 - (2 * statistic) / (n1 * n2)
        
        # Confidence interval (approximation)
        z_critical = stats.norm.ppf(1 - alpha / 2)
        se = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
        
        ci_low = r - z_critical * (1 / se)
        ci_high = r + z_critical * (1 / se)
        
        significant = p_value < alpha
        
        return StatisticalTest(
            p_value=float(p_value),
            effect_size=float(r),
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            significant=significant
        )
    
    def paired_t_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        alpha: float = 0.05
    ) -> StatisticalTest:
        """
        Perform paired t-test (parametric alternative to Wilcoxon).
        
        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B
            alpha: Significance level
            
        Returns:
            StatisticalTest object
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Scores must have same length for paired test")
        
        # Paired t-test
        statistic, p_value = stats.ttest_rel(scores_a, scores_b)
        
        # Cohen's d effect size
        differences = np.array(scores_a) - np.array(scores_b)
        d = np.mean(differences) / np.std(differences, ddof=1)
        
        # Confidence interval
        n = len(scores_a)
        se = stats.sem(differences)
        ci = stats.t.interval(1 - alpha, n - 1, loc=np.mean(differences), scale=se)
        
        significant = p_value < alpha
        
        return StatisticalTest(
            p_value=float(p_value),
            effect_size=float(d),
            ci_low=float(ci[0]),
            ci_high=float(ci[1]),
            significant=significant
        )
    
    def compare_methods(
        self,
        method_scores: dict,
        test_type: str = 'wilcoxon',
        alpha: float = 0.05
    ) -> dict:
        """
        Compare multiple methods with statistical tests.
        
        Args:
            method_scores: Dict mapping method names to score lists
            test_type: 'wilcoxon', 'mann_whitney', or 't_test'
            alpha: Significance level
            
        Returns:
            Dictionary of pairwise test results
        """
        methods = list(method_scores.keys())
        results = {}
        
        for i, method_a in enumerate(methods):
            for method_b in methods[i + 1:]:
                scores_a = method_scores[method_a]
                scores_b = method_scores[method_b]
                
                pair_name = f"{method_a}_vs_{method_b}"
                
                if test_type == 'wilcoxon':
                    test_result = self.wilcoxon_test(scores_a, scores_b, alpha)
                elif test_type == 'mann_whitney':
                    test_result = self.mann_whitney_test(scores_a, scores_b, alpha)
                elif test_type == 't_test':
                    test_result = self.paired_t_test(scores_a, scores_b, alpha)
                else:
                    raise ValueError(f"Unknown test type: {test_type}")
                
                results[pair_name] = test_result.to_dict()
        
        return results
