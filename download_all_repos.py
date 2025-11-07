#!/usr/bin/env python3
"""
Convenience script to download all repository categories at once.
Downloads: Neovim, Lua, Bash, Zsh, Python, and Ethical Hacking repositories.
"""
import sys
from pathlib import Path

# Import the download function
sys.path.insert(0, str(Path(__file__).parent))
from download_repos import download_repos

def main():
    print("🚀 SheepOp - Downloading All Repository Categories")
    print("=" * 60)
    print("\nThis will download:")
    print("  📦 Neovim configurations and plugins")
    print("  📦 Lua programming repositories")
    print("  📦 Bash/shell script repositories")
    print("  📦 Zsh configuration and plugins")
    print("  📦 Python programming repositories")
    print("  📦 Ethical hacking and cybersecurity tools")
    print("\n" + "=" * 60)
    
    # Default settings
    categories = ['nvim', 'lua', 'bash', 'zsh', 'python', 'hacking']
    max_repos_per_category = 50
    min_stars = 100
    output_dir = "data/repos"
    shallow = True  # Default to shallow clones
    max_size_gb = 1024.0  # Default 1 TB
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if '--help' in sys.argv or '-h' in sys.argv:
            print("\nUsage:")
            print("  python3 download_all_repos.py [options]")
            print("\nOptions:")
            print("  --max-repos N    Maximum repos per category (default: 50)")
            print("  --min-stars N    Minimum stars (default: 100)")
            print("  --output DIR     Output directory (default: data/repos)")
            print("  --max-size N     Maximum total size in GB (default: 1024.0 = 1 TB)")
            print("  --full-clone     Do full clone instead of shallow")
            print("\nExample:")
            print("  python3 download_all_repos.py --max-repos 100 --min-stars 200 --max-size 1024.0")
            return
        
        # Parse simple arguments
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            if args[i] == '--max-repos' and i + 1 < len(args):
                max_repos_per_category = int(args[i + 1])
                i += 2
            elif args[i] == '--min-stars' and i + 1 < len(args):
                min_stars = int(args[i + 1])
                i += 2
            elif args[i] == '--output' and i + 1 < len(args):
                output_dir = args[i + 1]
                i += 2
            elif args[i] == '--max-size' and i + 1 < len(args):
                max_size_gb = float(args[i + 1])
                i += 2
            elif args[i] == '--full-clone':
                shallow = False
                i += 1
            else:
                i += 1
    
    print(f"\n📊 Settings:")
    print(f"   Categories: {', '.join(categories)}")
    print(f"   Max repos per category: {max_repos_per_category}")
    print(f"   Min stars: {min_stars}")
    print(f"   Output directory: {output_dir}")
    print(f"   Max size: {max_size_gb} GB ({max_size_gb / 1024.0:.2f} TB)")
    print(f"   Shallow clone: {shallow}")
    print()
    
    # Confirm before starting
    try:
        response = input("Continue? [Y/n]: ").strip().lower()
        if response and response != 'y':
            print("Cancelled.")
            return
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    # Download all categories
    success = download_repos(
        output_dir=output_dir,
        license='mit',  # Default to MIT license
        min_stars=min_stars,
        max_repos=max_repos_per_category,
        shallow=shallow,
        categories=categories,
        max_size_gb=max_size_gb,
    )
    
    if success:
        print(f"\n🎉 All downloads complete!")
        print(f"\n📚 You can now train with:")
        print(f"   python3 train.py --data data/ --config config.json --device cuda")
        print(f"\n   This will process:")
        print(f"   - Your existing 196 GB of text data")
        print(f"   - All downloaded code repositories")
    else:
        print("\n❌ Some downloads failed. Check the output above for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()

