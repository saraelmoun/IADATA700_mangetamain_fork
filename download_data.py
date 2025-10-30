"""
Download data files from AWS S3 if not present locally.

This script checks if the required CSV files exist in the data/ directory.
If not, it downloads them from the public S3 bucket.
"""

from pathlib import Path
import urllib.request
import sys


# S3 URLs for the data files
S3_URLS = {
    "RAW_recipes.csv": "https://iadata700-mangetamain-data.s3.eu-west-3.amazonaws.com/RAW_recipes+2.csv",
    "RAW_interactions.csv": "https://iadata700-mangetamain-data.s3.eu-west-3.amazonaws.com/RAW_interactions+3.csv",
}

# Local data directory
DATA_DIR = Path("data")


def download_file(url: str, destination: Path) -> bool:
    """
    Download a file from a URL to a local destination with optimizations.
    Args:
        url: The URL to download from
        destination: The local path to save the file
    Returns:
        True if download successful, False otherwise
    """
    try:
        print(f"📥 Downloading {destination.name}...")
        print(f"🔗 URL: {url}")

        # Create request with better headers and timeout
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Python-urllib/3.12 MangetamainApp/1.0",
                "Accept": "*/*",
                "Accept-Encoding": "identity",  # Disable compression for speed
            },
        )

        # Open with timeout
        with urllib.request.urlopen(req, timeout=30) as response:
            file_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks instead of 8KB
            last_progress = -1  # Track last shown progress to reduce prints

            print(f"📦 Size: {file_size / (1024 * 1024):.1f} MB")

            with open(destination, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Show progress less frequently (every 10%)
                    if file_size > 0:
                        progress = int((downloaded / file_size) * 100)
                        if progress >= last_progress + 10:  # Only print every 10%
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = file_size / (1024 * 1024)
                            print(f"⏳ Progress: {progress}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                            last_progress = progress

        print(f"✅ Successfully downloaded {destination.name}")
        return True

    except urllib.request.URLError as e:
        print(f"❌ Network error downloading {destination.name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error downloading {destination.name}: {e}")
        return False


def ensure_data_files():
    """
    Ensure all required data files are present.
    Downloads from S3 if missing.
    """
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("Checking data files...")
    print("=" * 60)
    all_present = True
    files_to_download = []
    # Check which files are missing
    for filename in S3_URLS.keys():
        filepath = DATA_DIR / filename
        if filepath.exists():
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} exists ({file_size_mb:.1f} MB)")
        else:
            print(f"❌ {filename} missing")
            files_to_download.append(filename)
            all_present = False
    if all_present:
        print("\n✅ All data files are present!")
        return True
    # Download missing files
    print(f"\n📥 Downloading {len(files_to_download)} missing file(s)...")
    print("=" * 60)
    success = True
    for filename in files_to_download:
        url = S3_URLS[filename]
        destination = DATA_DIR / filename
        if not download_file(url, destination):
            success = False
            print(f"⚠️  Failed to download {filename}")
    print("=" * 60)
    if success:
        print("✅ All files downloaded successfully!")
    else:
        print("⚠️  Some files failed to download. Please check your internet connection.")
        sys.exit(1)
    return success


if __name__ == "__main__":
    ensure_data_files()
