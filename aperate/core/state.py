"""Pipeline state management for aperate."""

from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime


class PipelineStep(Enum):
    """Enumeration of pipeline steps."""
    INIT = "init"
    PSF = "psf"
    HOMOGENIZE = "homogenize"
    DETECT = "detect"
    PHOTOMETRY = "photometry"
    POSTPROCESS = "postprocess"


class StepStatus(Enum):
    """Status of a pipeline step."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineState:
    """Manage pipeline execution state."""
    
    def __init__(self):
        self.steps: Dict[PipelineStep, Dict[str, Any]] = {}
        self._initialize_steps()
    
    def _initialize_steps(self):
        """Initialize all steps as not started."""
        for step in PipelineStep:
            self.steps[step] = {
                "status": StepStatus.NOT_STARTED,
                "timestamp": None,
                "metadata": {}
            }
    
    def update_step(
        self, 
        step: PipelineStep, 
        status: StepStatus, 
        **metadata
    ) -> None:
        """Update the status and metadata for a pipeline step."""
        self.steps[step]["status"] = status
        self.steps[step]["timestamp"] = datetime.now().isoformat()
        self.steps[step]["metadata"].update(metadata)
    
    def get_step_status(self, step: PipelineStep) -> StepStatus:
        """Get the current status of a pipeline step."""
        return self.steps[step]["status"]
    
    def is_completed(self, step: PipelineStep) -> bool:
        """Check if a step is completed."""
        return self.get_step_status(step) == StepStatus.COMPLETED
    
    def can_run_step(self, step: PipelineStep) -> bool:
        """Check if a step can be run based on dependencies."""
        # TODO: Implement dependency checking logic
        # This is placeholder - user will define actual dependencies
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            step.value: {
                "status": info["status"].value,
                "timestamp": info["timestamp"],
                "metadata": info["metadata"]
            }
            for step, info in self.steps.items()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        """Create state from dictionary."""
        state = cls()
        for step_name, step_info in data.items():
            step = PipelineStep(step_name)
            state.steps[step] = {
                "status": StepStatus(step_info["status"]),
                "timestamp": step_info["timestamp"],
                "metadata": step_info["metadata"]
            }
        return state