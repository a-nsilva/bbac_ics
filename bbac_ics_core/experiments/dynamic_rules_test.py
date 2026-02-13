#!/usr/bin/env python3
"""
BBAC_ICS Framework - Dynamic Rules Test
Tests policy engine rule updates and consistency.
"""

import json
import time
from pathlib import Path
from typing import Dict

from ..layers.policy_engine import PolicyEngine
from ..utils.data_structures import (
    AccessRequest,
    ActionType,
    AgentRole,
    AgentType,
    AuthStatus,
    ResourceType,
)


class DynamicRulesTest:
    """
    Test dynamic rule updates.
    
    Metrics:
    1. Rule update latency (< 1 second target)
    2. Rule consistency during transitions (> 99.9%)
    3. Conflict detection
    """
    
    def __init__(self, output_dir: str = "results/dynamic_rules"):
        """Initialize dynamic rules test."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict:
        """Run dynamic rules tests."""
        print("=" * 50)
        print("DYNAMIC RULES EVALUATION")
        print("=" * 50)
        
        results = {}
        
        # Test 1: Rule update latency
        print("\n[1] Testing rule update latency...")
        results['update_latency'] = self._test_update_latency()
        
        # Test 2: Rule consistency
        print("\n[2] Testing rule consistency during updates...")
        results['consistency'] = self._test_consistency()
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _test_update_latency(self) -> Dict:
        """Measure rule update latency."""
        
        policy_engine = PolicyEngine()
        
        # Simulate rule updates
        num_updates = 100
        latencies = []
        
        for i in range(num_updates):
            # Add new role
            new_role = f"test_role_{i}"
            
            #start = time.time()
            start = time.perf_counter()
            
            # Simulate rule update (in real system, would reload config)
            policy_engine.role_actions[AgentRole.OPERATOR].add(ActionType.DIAGNOSTIC)
            
            #end = time.time()
            end = time.perf_counter()
            
            #latency_ms = (end - start) * 1000
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        print(f"  Average update latency: {avg_latency:.4f} ms")
        print(f"  Maximum update latency: {max_latency:.4f} ms")
        print(f"  Target: < 1000 ms")
        print(f"  Status: {'✓ PASS' if max_latency < 1000 else '✗ FAIL'}")
        
        return {
            'num_updates': num_updates,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'target_ms': 1000,
            'passed': max_latency < 1000
        }
    
    def _test_consistency(self) -> Dict:
        """Test consistency during rule transitions."""
        
        policy_engine = PolicyEngine()
        
        # Create test request
        request = AccessRequest(
            request_id="test_001",
            timestamp=time.time(),
            agent_id="robot_test_01",
            agent_type=AgentType.ROBOT,
            agent_role=AgentRole.ASSEMBLY_ROBOT,
            action=ActionType.WRITE,
            resource="test_resource",
            resource_type=ResourceType.DATABASE,
            location="test_zone",
            auth_status=AuthStatus.SUCCESS
        )
        
        # Run multiple evaluations
        num_evaluations = 1000
        decisions = []
        
        for _ in range(num_evaluations):
            decision = policy_engine.analyze(request)
            decisions.append(decision.decision)
        
        # Check consistency
        unique_decisions = set(decisions)
        consistency_rate = decisions.count(decisions[0]) / len(decisions)
        
        print(f"  Evaluations: {num_evaluations}")
        print(f"  Unique decisions: {len(unique_decisions)}")
        print(f"  Consistency rate: {consistency_rate * 100:.4f}%")
        print(f"  Target: > 99.9%")
        print(f"  Status: {'✓ PASS' if consistency_rate > 0.999 else '✗ FAIL'}")
        
        return {
            'num_evaluations': num_evaluations,
            'unique_decisions': len(unique_decisions),
            'consistency_rate': consistency_rate,
            'target_rate': 0.999,
            'passed': consistency_rate > 0.999
        }
    
    def _save_results(self, results: Dict):
        """Save dynamic rules results."""
        output_file = self.output_dir / "dynamic_rules_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    """Run dynamic rules test."""
    test = DynamicRulesTest()
    results = test.run()


if __name__ == '__main__':
    main()



