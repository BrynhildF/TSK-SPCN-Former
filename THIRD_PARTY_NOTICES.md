# Source and third-party notices

`model.py` is copied without modification from
`minimal_tsk_spcn/model.py` in the source project. The model implementation
uses the PyTorch API; PyTorch is an external dependency and is distributed
under its own license.

The source project's MIT license is retained in `LICENSE`.

This model-only package does not distribute `spot.py`, `pot.py`, or the
SPOT implementation attributed to Alban Siffer / Amossys. The SPOT notice
in the full implementation is therefore not a claim that SPOT is included
in this package.

This notice describes the files included here; it is not an independent
audit of all upstream attribution or licensing requirements.

The bundled GWD and SWaT weight files are model states exported from this
project's existing experiments. They do not include the underlying dataset
files. Their dataset scope, selection record and checksums are documented
in `weights/manifest.json`; no dataset license grant is implied.
