#!/usr/bin/env python3
"""
Timeline generation framework for GSAT-6A analysis.

Converts raw detection events and telemetry into human-readable timeline output.
The timeline is data-driven: it only reports what was actually detected/measured,
without editorial explanations in the simulation code.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class EventSeverity(Enum):
    """Event severity levels."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2


@dataclass
class TimelineEvent:
    """A single event in the failure timeline."""
    time_seconds: float
    severity: EventSeverity
    event_type: str  # "causal_detection", "threshold_alert", "telemetry_milestone"
    subsystem: str
    message: str
    confidence: Optional[float] = None


class Timeline:
    """Timeline builder that generates output from detected events."""
    
    def __init__(self):
        """Initialize timeline."""
        self.events: List[TimelineEvent] = []
    
    def add_event(self, time_seconds: float, severity: EventSeverity, 
                  event_type: str, subsystem: str, message: str, 
                  confidence: Optional[float] = None):
        """Add an event to the timeline."""
        self.events.append(TimelineEvent(
            time_seconds=time_seconds,
            severity=severity,
            event_type=event_type,
            subsystem=subsystem,
            message=message,
            confidence=confidence
        ))
    
    def sort(self):
        """Sort events by time."""
        self.events.sort(key=lambda e: e.time_seconds)
    
    def print_timeline(self):
        """Print the timeline in a readable format."""
        if not self.events:
            print("No events recorded")
            return
        
        self.sort()
        
        print("="*80)
        print("TIMELINE")
        print("="*80)
        
        for event in self.events:
            severity_icon = {
                EventSeverity.INFO: "ℹ",
                EventSeverity.WARNING: "⚠",
                EventSeverity.CRITICAL: "🔴"
            }[event.severity]
            
            time_str = f"T+{event.time_seconds:7.1f}s"
            
            if event.confidence is not None:
                conf_str = f" ({event.confidence:.0%})"
            else:
                conf_str = ""
            
            print(f"{severity_icon} {time_str} [{event.subsystem:12s}] {event.message}{conf_str}")
        
        print("="*80 + "\n")
    
    def get_first_detection(self, event_type: str) -> Optional[float]:
        """Get the time of the first detection of a specific type."""
        for event in sorted(self.events, key=lambda e: e.time_seconds):
            if event.event_type == event_type:
                return event.time_seconds
        return None
    
    def get_lead_time(self, first_type: str, second_type: str) -> Optional[float]:
        """Calculate lead time between two detection types."""
        first = self.get_first_detection(first_type)
        second = self.get_first_detection(second_type)
        
        if first is not None and second is not None:
            return second - first
        return None
