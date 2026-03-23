#!/usr/bin/env python3
"""
Download and Prepare Arabic NLP Benchmark Datasets
====================================================
Downloads and prepares all 11 Arabic NLP benchmarks used for
cross-domain validation of ICL prompt-design findings.

Categories:
  - Dialect Identification (2 datasets)
  - Sentiment Analysis (7 datasets)
  - Content Moderation (2 datasets)

Usage:
    python3 01_download_benchmarks.py

Output:
    data/<dataset_name>/ for each dataset
"""

import os
import json
import subprocess

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def clone_repo(url, target_dir):
    """Clone a git repository."""
    if os.path.exists(target_dir):
        print(f"  Already exists: {target_dir}")
        return True
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, target_dir],
                       check=True, capture_output=True)
        print(f"  Cloned: {target_dir}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  ERROR: Failed to clone {url}")
        return False


def write_readme(directory, name, reference, instructions):
    """Write a README with download instructions."""
    os.makedirs(directory, exist_ok=True)
    readme = os.path.join(directory, "README.md")
    with open(readme, "w") as f:
        f.write(f"# {name}\n\n")
        f.write(f"**Reference:** {reference}\n\n")
        f.write(f"**How to obtain:**\n{instructions}\n")


def setup_nadi():
    """NADI 2020 — Nuanced Arabic Dialect Identification."""
    print("\n[1/11] NADI 2020 (Dialect Identification)")
    nadi_dir = os.path.join(DATA_DIR, "nadi_2020")
    os.makedirs(nadi_dir, exist_ok=True)

    local_nadi = os.path.expanduser("~/sudanese_dialect_project/data/raw/NADI")
    if os.path.exists(local_nadi):
        print(f"  Found local copy at: {local_nadi}")
        for f in ["train.json", "dev.json", "test.json", "labels.json"]:
            src = os.path.join(local_nadi, f)
            dst = os.path.join(nadi_dir, f)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)
        return True

    print("  MANUAL: Download from https://github.com/Wikipedia-based-ADI")
    return False


def setup_madar():
    """MADAR-26 — Arabic Fine-Grained Dialect Identification."""
    print("\n[2/11] MADAR-26 (Dialect Identification)")
    write_readme(
        os.path.join(DATA_DIR, "madar_26"),
        "MADAR-26", "Bouamor et al. (2019)",
        "Request from: https://camel.abudhabi.nyu.edu/madar-shared-task-2019/")
    return False


def setup_astd():
    """ASTD — Arabic Sentiment Tweets Dataset."""
    print("\n[3/11] ASTD (Sentiment Analysis)")
    return clone_repo("https://github.com/mahmoudnabil/ASTD.git",
                       os.path.join(DATA_DIR, "astd", "repo"))


def setup_arsas():
    """ArSAS — Arabic Speech-Act and Sentiment."""
    print("\n[4/11] ArSAS (Sentiment Analysis)")
    write_readme(
        os.path.join(DATA_DIR, "arsas"),
        "ArSAS", "AbdelRahim & Magdy (2018)",
        "Request from the authors or https://alt.qcri.org/resources/")
    return False


def setup_asad():
    """ASAD — Arabic Sentiment Analysis Dataset."""
    print("\n[5/11] ASAD (Sentiment Analysis)")
    return clone_repo("https://github.com/basharalhafni/ASAD.git",
                       os.path.join(DATA_DIR, "asad", "repo"))


def setup_labr():
    """LABR — Large-scale Arabic Book Reviews."""
    print("\n[6/11] LABR (Sentiment Analysis)")
    return clone_repo("https://github.com/mohamedadaly/LABR.git",
                       os.path.join(DATA_DIR, "labr", "repo"))


def setup_semeval2016():
    """SemEval-2016 Task 7 — Sentiment Intensity."""
    print("\n[7/11] SemEval-2016 Task 7 (Sentiment Analysis)")
    write_readme(
        os.path.join(DATA_DIR, "semeval_2016"),
        "SemEval-2016 Task 7", "Kiritchenko et al. (2016)",
        "Download from: https://alt.qcri.org/semeval2016/task7/")
    return False


def setup_alomari():
    """Alomari 2017 — Arabic Tweets Sentiment."""
    print("\n[8/11] Alomari 2017 (Sentiment Analysis)")
    write_readme(
        os.path.join(DATA_DIR, "alomari_2017"),
        "Alomari 2017", "Alomari et al. (2017)",
        "Request from the authors")
    return False


def setup_arsentiment():
    """ArSentiment — Aggregated Arabic Sentiment."""
    print("\n[9/11] ArSentiment (Aggregated Sentiment)")
    write_readme(
        os.path.join(DATA_DIR, "arsentiment"),
        "ArSentiment", "Aggregated from ASTD, ArSAS, and others",
        "Constructed after individual datasets are available")
    return False


def setup_osact4():
    """OSACT4 — Offensive Language Detection."""
    print("\n[10/11] OSACT4 (Content Moderation)")
    write_readme(
        os.path.join(DATA_DIR, "osact4"),
        "OSACT4", "Mubarak et al. (2020)",
        "Download from: https://edinburghnlp.inf.ed.ac.uk/workshops/OSACT4/")
    return False


def setup_adult_content():
    """Adult Content Detection on Arabic Twitter."""
    print("\n[11/11] Adult Content (Content Moderation)")
    write_readme(
        os.path.join(DATA_DIR, "adult_content"),
        "Adult Content Detection", "Mubarak et al. (2021)",
        "Request from the authors")
    return False


def main():
    print("=" * 60)
    print("Arabic NLP Benchmark Dataset Download")
    print("=" * 60)

    datasets = [
        ("NADI 2020", setup_nadi),
        ("MADAR-26", setup_madar),
        ("ASTD", setup_astd),
        ("ArSAS", setup_arsas),
        ("ASAD", setup_asad),
        ("LABR", setup_labr),
        ("SemEval-2016", setup_semeval2016),
        ("Alomari 2017", setup_alomari),
        ("ArSentiment", setup_arsentiment),
        ("OSACT4", setup_osact4),
        ("Adult Content", setup_adult_content),
    ]

    results = {}
    for name, func in datasets:
        results[name] = func()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "READY" if success else "MANUAL DOWNLOAD NEEDED"
        print(f"  [{status}] {name}")

    with open(os.path.join(DATA_DIR, "download_status.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
