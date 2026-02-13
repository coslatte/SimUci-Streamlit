"""Google Drive API service for downloading data files securely.

Uses a Google Cloud **service account** so that credentials are scoped
exclusively to this application — the user's personal Google account is
never exposed nor compromised.

Setup
-----
1. Create (or reuse) a Google Cloud project at https://console.cloud.google.com
2. Enable the **Google Drive API** for that project.
3. Create a **Service Account** → download the JSON key file.
4. Share the target Google Drive folder with the service account e-mail
   (e.g. ``simuci-data@your-project.iam.gserviceaccount.com``) as **Viewer**.
5. Copy the JSON key contents into ``.streamlit/secrets.toml`` under the
   ``[google_drive.service_account]`` section (see ``secrets.toml.example``).

Security notes
--------------
* The service account is a *separate identity* — it has **zero access** to any
  file that has not been explicitly shared with it.
* Credentials live in Streamlit secrets (never committed to the repo).
* Access can be revoked at any time by un-sharing the folder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDriveService:
    """Thin wrapper around the Google Drive v3 API using service-account auth."""

    def __init__(self, credentials_info: dict[str, Any]) -> None:
        """Initialise the service from a service-account JSON dict.

        Args:
            credentials_info: The parsed contents of the service-account JSON
                              key file (or the equivalent Streamlit secrets section).
        """
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

        creds = Credentials.from_service_account_info(
            credentials_info, scopes=_SCOPES
        )
        self._service = build(
            "drive", "v3", credentials=creds, cache_discovery=False
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def list_files(self, folder_id: str) -> list[dict[str, str]]:
        """Return a list of ``{id, name, mimeType}`` dicts for items in *folder_id*.

        Args:
            folder_id: The Google Drive folder ID to list.

        Returns:
            List of file metadata dictionaries.
        """
        query = f"'{folder_id}' in parents and trashed = false"
        results = (
            self._service.files()
            .list(q=query, fields="files(id, name, mimeType)", pageSize=100)
            .execute()
        )
        return results.get("files", [])

    def download_file(self, file_id: str, dest: Path) -> Path:
        """Download a file by its Drive ID to a local path.

        Uses ``MediaIoBaseDownload`` for reliable, chunked transfers that
        work for files of any size.

        Args:
            file_id: The Google Drive file ID.
            dest: Local destination path.

        Returns:
            The destination ``Path`` on success.
        """
        from googleapiclient.http import MediaIoBaseDownload

        request = self._service.files().get_media(fileId=file_id)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        logger.info("Downloaded %s → %s", file_id, dest)
        return dest

    def download_file_by_name(
        self,
        folder_id: str,
        filename: str,
        dest: Path,
    ) -> Path | None:
        """Find *filename* inside *folder_id* and download it to *dest*.

        Args:
            folder_id: The Google Drive folder ID to search in.
            filename: The exact file name to match.
            dest: Local destination path.

        Returns:
            The destination ``Path`` on success, or ``None`` if the file
            was not found in the folder.
        """
        files = self.list_files(folder_id)
        for f in files:
            if f["name"] == filename:
                return self.download_file(f["id"], dest)

        logger.warning("File %r not found in folder %s", filename, folder_id)
        return None


# ------------------------------------------------------------------
# Cached singleton
# ------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_drive_service() -> GoogleDriveService | None:
    """Build and cache a :class:`GoogleDriveService` from Streamlit secrets.

    Returns ``None`` when:
    * The ``[google_drive.service_account]`` section is missing in secrets.
    * The ``google-api-python-client`` / ``google-auth`` packages are not installed.
    * Authentication fails for any reason.
    """
    try:
        gd_section = st.secrets.get("google_drive", {})
        sa_info: dict[str, Any] | None = gd_section.get("service_account")
        if not sa_info:
            return None
        # Streamlit AttrDict → plain dict for google-auth
        return GoogleDriveService(dict(sa_info))
    except ImportError:
        logger.warning(
            "Google API libraries not installed — "
            "Google Drive service account auth unavailable."
        )
        return None
    except Exception:
        logger.exception("Failed to initialise Google Drive service")
        return None
