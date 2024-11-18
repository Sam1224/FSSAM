# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import hydra
from hydra import initialize_config_module

hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize_config_module('sam2_configs', version_base='1.2')
