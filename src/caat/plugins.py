"""The component catalog used by the CLI and the contract suite."""

from .adapters import ADAPTERS
from .calibrate import CALIBRATORS
from .features import FEATURES
from .metrics import METRICS
from .renderers import RENDERERS
from .rules import RULES


STAGES = {
    "features": (FEATURES, ("direction", "requires", "cost")),
    "calibrators": (CALIBRATORS, ("monotone", "input_scale")),
    "metrics": (METRICS, ("requires_probability", "direction")),
    "rules": (RULES, ("assumptions",)),
    "adapters": (ADAPTERS, ("source_format", "source_version")),
    "renderers": (RENDERERS, ("format", "commands")),
}


def catalog():
    return {
        stage: [
            {
                "name": name,
                "source": registry.source(name),
                **{key: getattr(component, key) for key in fields},
            }
            for name, component in registry.items()
        ]
        for stage, (registry, fields) in STAGES.items()
    }
