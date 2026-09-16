"""The `gcg` product package.

This file exists for ONE reason and it is not style: `modal run gcg/modal_app.py` puts
`paper-evals/gcg/` on `sys.path` ahead of everything, where the module `gcg.py` would shadow the
package `gcg/` and `from gcg import gcg` resolves to the module itself (PEP 420: a regular module
found on a later path entry beats a namespace portion found on an earlier one). With an
`__init__.py` the package is regular and wins at the first path entry, which is `paper-evals/`.
"""
