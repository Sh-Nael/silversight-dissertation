"""SilverSight web API (Track U). A thin FastAPI layer over the ``xaglab`` library.

The API contains no research logic: every endpoint calls the same library functions
the CLI and the experiments use, so the web app can never show a number the command
line cannot reproduce.
"""

from xaglab.api.app import create_app

__all__ = ["create_app"]
