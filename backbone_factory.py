from typing import Callable, Dict
from dataclasses import dataclass
import inspect

from classification_models.resnet import resnet
from classification_models.efficientnet import efficientnet
from classification_models.resnet import SUBTYPE_CONFIG as SUBTYPES_RESNET
from classification_models.efficientnet import SUBTYPE_CONFIG as SUBTYPES_EFFICIENTNET


###############################################################################
# Defaults.
###############################################################################
ARCHITECTURES = (
    *list(SUBTYPES_RESNET.keys()),
    *list(SUBTYPES_EFFICIENTNET.keys()),
)

###############################################################################
# Custom errors.
###############################################################################
class InvalidBackbone(Exception):
    def __init__(self, name):
        self.name = name
        self.msn = (
            f"Invalid backbone architecture `{self.name}`.\n"
            f"Valid architectures are: \n"
            f"{ARCHITECTURES}\n"
        )
        super().__init__(self.msn)


class InvalidConfig(Exception):
    def __init__(self, invalid, expected):
        self.invalid = invalid
        self.expected = expected
        self.msn = (
            f"Invalid `config` parameters: `{self.invalid}`.\n"
            f"Valid parameters are: \n"
            f"{self.expected}\n"
        )
        super().__init__(self.msn)


###############################################################################
# Builder.
###############################################################################
@dataclass
class BackboneFactory:
    arch: str
    config: Dict = None

    def __post_init__(self):
        if self.config is None:
            self.config = dict()

        # Verify backbone architecture exists.
        if not self.arch in ARCHITECTURES:
            raise InvalidBackbone(self.arch)

    def _validate_config(self, func: Callable) -> bool:
        func_args = list(
            inspect.Signature().from_callable(func).parameters.keys()
        )
        invalid = []
        for param in self.config.keys():
            if not param in list(func_args):
                invalid.append(param)
        if len(invalid) > 0:
            raise InvalidConfig(invalid, func_args)

    def _init_backbone(self, func: Callable):
        self._validate_config(func)

        return func(self.arch, **self.config)

    def get(self):
        if self.arch in SUBTYPES_RESNET.keys():
            arch = resnet
            # backbone = self._init_backbone(resnet)
        elif self.arch in SUBTYPES_EFFICIENTNET.keys():
            arch = efficientnet
            # backbone = self._init_backbone(efficientnet)
        else:
            raise InvalidBackbone(self.arch)

        # return backbone
        return self._init_backbone(arch)
