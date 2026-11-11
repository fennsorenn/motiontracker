"""Repository and data locations. MT_DATA overrides the gitignored data/ folder."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("MT_DATA", REPO_ROOT / "data"))
MHR_ASSETS = DATA / "mhr" / "assets"
CMU_ROOT = DATA / "cmu" / "all_asfamc" / "subjects"
# VRoid sample avatars of the sibling volleyball project (license register: only the unsuffixed, permissive ones)
AVATARS = REPO_ROOT.parent / "volleyball" / "tools" / "avatars" / "models"
