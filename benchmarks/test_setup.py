"""
Test script to verify benchmark setup.

Run this to check that all dependencies are available.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        import numpy as np

        print("✓ numpy")
    except ImportError:
        print("✗ numpy - install with: pip install numpy")
        return False

    try:
        import xxhash

        print("✓ xxhash")
    except ImportError:
        print("✗ xxhash - install with: pip install xxhash")
        return False

    try:
        from tqdm import tqdm

        print("✓ tqdm")
    except ImportError:
        print("✗ tqdm - install with: pip install tqdm")
        return False

    try:
        from dotenv import load_dotenv

        print("✓ python-dotenv")
    except ImportError:
        print("✗ python-dotenv - install with: pip install python-dotenv")
        return False

    try:
        from ragdoll import Ragdoll

        print("✓ ragdoll")
    except ImportError:
        print("✗ ragdoll - ensure you're in the RAGdoll directory")
        return False

    return True


def test_directory_structure():
    """Test that benchmark directories exist."""
    print("\nTesting directory structure...")

    benchmark_dir = Path(__file__).parent

    required_dirs = [
        benchmark_dir / "datasets",
        benchmark_dir / "results",
        benchmark_dir / "db",
    ]

    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✓ {dir_path.name}/")
        else:
            print(f"✗ {dir_path.name}/ - directory missing")
            return False

    return True


def test_api_key():
    """Test that OPENAI_API_KEY is set."""
    print("\nTesting API key...")

    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        print(f"✓ OPENAI_API_KEY is set (length: {len(api_key)})")
        return True
    else:
        print("✗ OPENAI_API_KEY not found")
        print("  Set it in .env or environment variables")
        return False


def test_dataset_availability():
    """Test that datasets are available."""
    print("\nChecking datasets...")

    datasets_dir = Path(__file__).parent / "datasets"

    datasets = {
        "2wikimultihopqa": datasets_dir / "2wikimultihopqa.json",
    }

    found_any = False
    for name, path in datasets.items():
        if path.exists():
            print(f"✓ {name} dataset found")
            found_any = True
        else:
            print(f"⚠ {name} dataset not found at: {path}")
            print(
                f"  Download from: https://github.com/circlemind-ai/fast-graphrag/tree/main/benchmarks/datasets"
            )

    return found_any


def main():
    """Run all tests."""
    print("=" * 70)
    print("RAGdoll Benchmark Setup Test")
    print("=" * 70)
    print()

    tests = [
        ("Imports", test_imports),
        ("Directory Structure", test_directory_structure),
        ("API Key", test_api_key),
        ("Datasets", test_dataset_availability),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} - Error: {e}")
            results.append((name, False))
        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")

    print()
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("\n✅ All tests passed! Ready to run benchmarks.")
        print("\nNext steps:")
        print("  1. Ensure datasets are downloaded (if needed)")
        print("  2. Run: .\\run_benchmarks.ps1 -Subset 51")
    else:
        print("\n⚠️  Some tests failed. Please address the issues above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
