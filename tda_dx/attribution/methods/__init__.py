from tda_dx.attribution.methods.base import AttributionMethod
from tda_dx.attribution.methods.factgrass import FactGraSSMethod
from tda_dx.attribution.methods.grad_cos import GradCosMethod
from tda_dx.attribution.methods.logra import LoGraMethod
from tda_dx.attribution.methods.trak import TRAKMethod

METHOD_REGISTRY = {
    'factgrass': FactGraSSMethod,
    'grad_cos': GradCosMethod,
    'logra': LoGraMethod,
    'trak': TRAKMethod,
}

__all__ = [
    'AttributionMethod',
    'FactGraSSMethod',
    'GradCosMethod',
    'LoGraMethod',
    'TRAKMethod',
    'METHOD_REGISTRY',
]
