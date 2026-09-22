"""The `autointerp` product package (the Delphi-style SAE autointerp evaluation).

Like `gcg/__init__.py` this file is here so that `modal run autointerp/modal_app.py`, which puts
`paper-evals/autointerp/` on `sys.path` ahead of `paper-evals/`, still resolves `autointerp.build`
to the package's module rather than to a shadowing top-level one.
"""
