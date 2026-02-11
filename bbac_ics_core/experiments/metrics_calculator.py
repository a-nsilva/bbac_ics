#!/usr/bin/env python3
"""
BBAC ICS Framework - Metrics Calculator
Centralized metrics computation for experiments.
"""
import numpy as np
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
from typing import List

from ..utils.data_structures import (
    ClassificationMetrics,
    LatencyMetrics,
    PerformanceMetrics
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
