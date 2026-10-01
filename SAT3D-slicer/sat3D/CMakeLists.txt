#-----------------------------------------------------------------------------
set(MODULE_NAME sat3D)

#-----------------------------------------------------------------------------
# Ship the module script plus every Python package it imports at runtime.
file(GLOB_RECURSE MODULE_PACKAGE_SCRIPTS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  ${MODULE_NAME}Lib/*.py
  networks/*.py
  segment_anything_with_swin_conf_plus/*.py
  )

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_PACKAGE_SCRIPTS}
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/TextEncoder/config.json
  Resources/TextEncoder/tokenizer_config.json
  Resources/TextEncoder/vocab.txt
  Resources/TextEncoder/LICENSE.md
  Resources/TextEncoder/README.md
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  )
