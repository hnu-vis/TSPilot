"""ReAct harness primitives.

The harness is intentionally split into:

* observation: model-visible state snapshots
* action space: valid next action constraints and tool cards
* state transition: runtime-owned state updates after tool output
"""

from core.harness.action_space import ActionSpaceBuilder, build_action_space
from core.harness.action_output import ActionOutputBuilder, ActionOutputBuildInput
from core.harness.capabilities import CapabilityRegistry, default_capability_registry
from core.harness.observation import ObservationFrame, build_observation_frame
from core.harness.observation_view import model_observation_view, public_observation_view
from core.harness.transition import StateTransitionEngine

__all__ = [
    "ActionSpaceBuilder",
    "ActionOutputBuildInput",
    "ActionOutputBuilder",
    "CapabilityRegistry",
    "ObservationFrame",
    "StateTransitionEngine",
    "build_action_space",
    "build_observation_frame",
    "default_capability_registry",
    "model_observation_view",
    "public_observation_view",
]
