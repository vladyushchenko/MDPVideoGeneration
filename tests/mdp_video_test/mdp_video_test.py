#
# Copyright (C) 2020, Vladyslav Yushchenko, iNTENCE automotive electronics GmbH
#
from types import ModuleType

import mdp_video


def test_pipeline_import() -> None:
    """
    Test if the import did return a module.
    """
    assert isinstance(mdp_video, ModuleType)
