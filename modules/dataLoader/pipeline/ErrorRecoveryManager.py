"""
ErrorRecoveryManager for centralized error handling and recovery coordination.

This module provides error classification, recovery strategy selection,
failure pattern detection, and automatic pipeline reconfiguration.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from .PipelineLogger import get_pipeline_logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    SKIP = "skip"
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    RECONFIGURE = "reconfigure"
    ABORT = "abort"


@dataclass
class FailurePattern:
    """Pattern of failures detected in the pipeline."""
    error_type: str
    module_name: str
    frequency: int
    time_window_seconds: float
    severity: ErrorSeverity
    suggested_action: str


@dataclass
class RecoveryAction:
    """Recovery action to be taken."""
    strategy: RecoveryStrategy
    parameters: Dict[str, Any]
    estimated_success_rate: float
    description: str


class ErrorRecoveryManager:
    """
    Centralized error handling and recovery coordination.
    
    Features:
    - Error classification and prioritization
    - Recovery strategy selection logic
    - Failure pattern detection
    - Automatic pipeline reconfiguration
    - Performance impact monitoring
    """
    
    def __init__(
        self,
        max_error_history: int = 1000,
        pattern_detection_window: float = 300.0,  # 5 minutes
        circuit_breaker_threshold: int = 10,
        recovery_timeout: float = 60.0
    ):
        self.max_error_history = max_error_history
        self.pattern_detection_window = pattern_detection_window
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.recovery_timeout = recovery_timeout
        
        # Error tracking
        self.error_history: deque = deque(maxlen=max_error_history)
        self.failure_patterns: List[FailurePattern] = []
        self.recovery_statistics = defaultdict(lambda: {
            'attempts': 0,
            'successes': 0,
            'failures': 0
        })
        
        # Circuit breaker states per module
        self.circuit_breakers = defaultdict(lambda: {
            'state': 'CLOSED',  # CLOSED, OPEN, HALF_OPEN
            'failure_count': 0,
            'last_failure_time': 0.0,
            'recovery_attempts': 0
        })
        
        # Recovery callbacks
        self.recovery_callbacks: Dict[str, Callable] = {}
        
        # Logger
        self.logger = get_pipeline_logger().get_module_logger("ErrorRecoveryManager")
        self.logger.info("ErrorRecoveryManager initialized")
    
    def classify_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify error type and determine severity."""
        error_str = str(error).lower()
        error_type_name = type(error).__name__
        
        # Classify error type
        if 'memory' in error_str or 'out of memory' in error_str:
            error_type = "RESOURCE_EXHAUSTION"
            severity = ErrorSeverity.HIGH
        elif 'connection' in error_str or 'network' in error_str or 'timeout' in error_str:
            error_type = "TRANSIENT"
            severity = ErrorSeverity.MEDIUM
        elif 'corrupt' in error_str or 'invalid' in error_str or 'malformed' in error_str:
            error_type = "DATA_CORRUPTION"
            severity = ErrorSeverity.MEDIUM
        elif 'config' in error_str or 'parameter' in error_str:
            error_type = "CONFIGURATION"
            severity = ErrorSeverity.HIGH
        elif 'system' in error_str or 'critical' in error_str:
            error_type = "CRITICAL"
            severity = ErrorSeverity.CRITICAL
        elif 'nonetype' in error_str and 'subscriptable' in error_str:
            error_type = "NULL_REFERENCE"
            severity = ErrorSeverity.HIGH
        else:
            error_type = "UNKNOWN"
            severity = ErrorSeverity.MEDIUM
        
        return {
            'error_type': error_type,
            'severity': severity,
            'error_class': error_type_name,
            'context': context,
            'timestamp': time.time()
        }
    
    def select_recovery_strategy(
        self, 
        error_classification: Dict[str, Any],
        module_name: str
    ) -> RecoveryAction:
        """Select appropriate recovery strategy based on error classification."""
        error_type = error_classification['error_type']
        severity = error_classification['severity']
        
        # Check circuit breaker state
        circuit_state = self.circuit_breakers[module_name]
        if circuit_state['state'] == 'OPEN':
            return RecoveryAction(
                strategy=RecoveryStrategy.CIRCUIT_BREAK,
                parameters={'wait_time': self.recovery_timeout},
                estimated_success_rate=0.1,
                description=f"Circuit breaker open for {module_name}"
            )
        
        # Strategy selection based on error type and severity
        if error_type == "TRANSIENT" and severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            return RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                parameters={'max_attempts': 3, 'backoff_factor': 2.0},
                estimated_success_rate=0.7,
                description="Retry with exponential backoff for transient error"
            )
        
        elif error_type == "DATA_CORRUPTION":
            return RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                parameters={'create_fallback': True},
                estimated_success_rate=0.9,
                description="Skip corrupted data and use fallback"
            )
        
        elif error_type == "NULL_REFERENCE":
            return RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                parameters={'create_safe_data': True},
                estimated_success_rate=0.8,
                description="Create safe fallback data for null reference"
            )
        
        elif error_type == "RESOURCE_EXHAUSTION":
            return RecoveryAction(
                strategy=RecoveryStrategy.RECONFIGURE,
                parameters={'reduce_batch_size': True, 'clear_cache': True},
                estimated_success_rate=0.6,
                description="Reconfigure pipeline to reduce resource usage"
            )
        
        elif severity == ErrorSeverity.CRITICAL:
            return RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                parameters={'save_state': True},
                estimated_success_rate=0.0,
                description="Abort processing due to critical error"
            )
        
        else:
            return RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                parameters={'create_safe_data': True},
                estimated_success_rate=0.5,
                description="Use fallback strategy for unknown error"
            )
    
    def should_retry_operation(
        self, 
        operation: str, 
        module_name: str,
        failure_count: int
    ) -> bool:
        """Determine if an operation should be retried."""
        # Check circuit breaker
        circuit_state = self.circuit_breakers[module_name]
        if circuit_state['state'] == 'OPEN':
            return False
        
        # Check failure count
        if failure_count >= 3:
            return False
        
        # Check recent failure rate
        recent_failures = self._get_recent_failures(module_name, 60.0)  # Last minute
        if len(recent_failures) > 5:
            return False
        
        return True
    
    def record_error(
        self,
        error: Exception,
        module_name: str,
        context: Dict[str, Any]
    ):
        """Record error and update circuit breaker state."""
        # Classify error
        classification = self.classify_error(error, context)
        classification['module_name'] = module_name
        
        # Store in history
        self.error_history.append(classification)
        
        # Update circuit breaker
        circuit_state = self.circuit_breakers[module_name]
        circuit_state['failure_count'] += 1
        circuit_state['last_failure_time'] = time.time()
        
        # Open circuit breaker if threshold exceeded
        if circuit_state['failure_count'] >= self.circuit_breaker_threshold:
            circuit_state['state'] = 'OPEN'
            self.logger.warning(f"Circuit breaker opened for {module_name}")
        
        # Log error
        self.logger.error(
            f"Error recorded: {classification['error_type']} in {module_name} "
            f"(severity: {classification['severity'].value})"
        )
        
        # Detect patterns
        self._detect_failure_patterns()
    
    def record_success(self, module_name: str, operation: str):
        """Record successful operation and update circuit breaker state."""
        circuit_state = self.circuit_breakers[module_name]
        
        # Reset failure count on success
        if circuit_state['state'] == 'HALF_OPEN':
            circuit_state['state'] = 'CLOSED'
            circuit_state['failure_count'] = 0
            self.logger.info(f"Circuit breaker closed for {module_name}")
        elif circuit_state['state'] == 'CLOSED':
            # Gradually reduce failure count
            circuit_state['failure_count'] = max(0, circuit_state['failure_count'] - 1)
    
    def _get_recent_failures(self, module_name: str, time_window: float) -> List[Dict[str, Any]]:
        """Get recent failures for a module within time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        return [
            error for error in self.error_history
            if (error.get('module_name') == module_name and 
                error.get('timestamp', 0) >= cutoff_time)
        ]
    
    def _detect_failure_patterns(self):
        """Detect patterns in failure history."""
        current_time = time.time()
        cutoff_time = current_time - self.pattern_detection_window
        
        # Group recent errors by type and module
        recent_errors = [
            error for error in self.error_history
            if error.get('timestamp', 0) >= cutoff_time
        ]
        
        error_groups = defaultdict(list)
        for error in recent_errors:
            key = (error.get('error_type'), error.get('module_name'))
            error_groups[key].append(error)
        
        # Detect patterns
        new_patterns = []
        for (error_type, module_name), errors in error_groups.items():
            if len(errors) >= 3:  # Pattern threshold
                severity = max(error.get('severity', ErrorSeverity.LOW) for error in errors)
                
                pattern = FailurePattern(
                    error_type=error_type,
                    module_name=module_name,
                    frequency=len(errors),
                    time_window_seconds=self.pattern_detection_window,
                    severity=severity,
                    suggested_action=self._suggest_pattern_action(error_type, len(errors))
                )
                new_patterns.append(pattern)
        
        # Update patterns
        self.failure_patterns = new_patterns
        
        if new_patterns:
            self.logger.warning(f"Detected {len(new_patterns)} failure patterns")
    
    def _suggest_pattern_action(self, error_type: str, frequency: int) -> str:
        """Suggest action based on failure pattern."""
        if frequency > 10:
            return "Consider disabling problematic module"
        elif error_type == "DATA_CORRUPTION":
            return "Validate input data sources"
        elif error_type == "RESOURCE_EXHAUSTION":
            return "Reduce batch size or enable memory optimization"
        elif error_type == "NULL_REFERENCE":
            return "Add comprehensive null checking"
        else:
            return "Monitor and investigate root cause"
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        return {
            'total_errors': len(self.error_history),
            'circuit_breaker_states': dict(self.circuit_breakers),
            'failure_patterns': [
                {
                    'error_type': p.error_type,
                    'module_name': p.module_name,
                    'frequency': p.frequency,
                    'severity': p.severity.value,
                    'suggested_action': p.suggested_action
                }
                for p in self.failure_patterns
            ],
            'recovery_stats': dict(self.recovery_statistics)
        }