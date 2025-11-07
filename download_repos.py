#!/usr/bin/env python3
"""
Download GitHub repositories with open licenses for code training.
Uses GitHub API to find and clone repositories automatically.
Includes support for Neovim, Lua, Bash, and ethical hacking repos.
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict
import json
import urllib.request
import urllib.parse
import time
from tqdm import tqdm


def get_directory_size(directory: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for entry in directory.rglob('*'):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except (OSError, PermissionError):
                    pass
    except Exception:
        pass
    return total

def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


# Open source licenses (permissive and commonly used)
OPEN_LICENSES = [
    'mit',
    'apache-2.0',
    'bsd-3-clause',
    'bsd-2-clause',
    'isc',
    'unlicense',
    'mpl-2.0',
    'lgpl-2.1',
    'lgpl-3.0',
    'gpl-2.0',
    'gpl-3.0',
]

# Popular programming languages
POPULAR_LANGUAGES = [
    'python',
    'javascript',
    'typescript',
    'java',
    'cpp',
    'c',
    'go',
    'rust',
    'ruby',
    'php',
    'swift',
    'kotlin',
    'scala',
    'r',
    'sql',
    'lua',
    'shell',  # For bash/shell scripts
]

# Predefined repository categories
REPO_CATEGORIES = {
    'nvim': {
        'query': 'neovim OR nvim-config OR neovim-config',
        'language': None,
        'description': 'Neovim configuration and plugins'
    },
    'lua': {
        'query': None,
        'language': 'lua',
        'description': 'Lua programming language repositories'
    },
    'bash': {
        'query': None,
        'language': 'shell',
        'description': 'Bash/shell script repositories'
    },
    'zsh': {
        'query': 'zsh-config OR oh-my-zsh OR zsh-plugin',
        'language': None,
        'description': 'Zsh configuration and plugins'
    },
    'python': {
        'query': None,
        'language': 'python',
        'description': 'Python programming repositories'
    },
    'hacking': {
        'query': 'ethical-hacking OR cybersecurity OR penetration-testing OR security-tools OR red-team',
        'language': None,
        'description': 'Ethical hacking and cybersecurity tools'
    },
    'security': {
        'query': 'security-tools OR cybersecurity OR penetration-testing OR red-team OR blue-team',
        'language': None,
        'description': 'Security and cybersecurity repositories'
    },
    'all-open': {
        'query': None,
        'language': None,
        'description': 'All repositories with open licenses (any language)'
    },
}


def search_github_repos(
    language: Optional[str] = None,
    license: Optional[str] = None,
    query: Optional[str] = None,
    min_stars: int = 100,
    max_repos: int = 100,
    sort: str = 'stars',
    order: str = 'desc'
) -> List[dict]:
    """
    Search GitHub for repositories matching criteria.
    
    Args:
        language: Programming language (e.g., 'python', 'javascript')
        license: License type (e.g., 'mit', 'apache-2.0')
        query: Custom search query
        min_stars: Minimum number of stars
        max_repos: Maximum number of repos to return
        sort: Sort by ('stars', 'updated', 'created')
        order: Order ('desc' or 'asc')
    
    Returns:
        List of repository dictionaries
    """
    # Build query
    query_parts = []
    
    if query:
        # Custom query (for categories like nvim, hacking)
        query_parts.append(query)
    else:
        # Standard language-based query
        if language:
            query_parts.append(f"language:{language}")
    
    if license:
        query_parts.append(f"license:{license}")
    query_parts.append(f"stars:>={min_stars}")
    
    search_query = " ".join(query_parts)
    
    # GitHub API endpoint
    base_url = "https://api.github.com/search/repositories"
    params = {
        'q': search_query,
        'sort': sort,
        'order': order,
        'per_page': min(100, max_repos),  # GitHub max is 100 per page
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    print(f"🔍 Searching GitHub for repositories...")
    print(f"   Query: {search_query}")
    print(f"   Max repos: {max_repos}")
    
    try:
        # Make request
        req = urllib.request.Request(url)
        req.add_header('Accept', 'application/vnd.github.v3+json')
        req.add_header('User-Agent', 'SheepOp-Repo-Downloader')
        
        # Add GitHub token if available
        github_token = os.environ.get('GITHUB_TOKEN')
        if github_token:
            req.add_header('Authorization', f'token {github_token}')
        
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
        
        repos = data.get('items', [])[:max_repos]
        print(f"✅ Found {len(repos)} repositories")
        return repos
    
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("❌ Rate limit exceeded. Please wait a few minutes or use a GitHub token.")
            print("   To use a token, set GITHUB_TOKEN environment variable:")
            print("   export GITHUB_TOKEN=your_token_here")
        else:
            print(f"❌ Error searching GitHub: {e}")
            if e.code == 422:
                print("   Tip: Try adjusting your search query or reducing max-repos")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def clone_repo(repo_url: str, output_dir: Path, depth: Optional[int] = None) -> bool:
    """
    Clone a repository.
    
    Args:
        repo_url: Repository URL (https://github.com/user/repo.git)
        output_dir: Directory to clone into
        depth: Shallow clone depth (None = full clone)
    
    Returns:
        True if successful
    """
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    target_dir = output_dir / repo_name
    
    # Skip if already exists
    if target_dir.exists():
        return True  # Silent skip (progress bar will show it)
    
    try:
        cmd = ['git', 'clone', '--quiet']  # Quiet mode for cleaner output
        if depth:
            cmd.extend(['--depth', str(depth)])
        cmd.append(repo_url)
        cmd.append(str(target_dir))
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        return result.returncode == 0
    
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        return False


def download_category(
    category: str,
    output_dir: Path,
    license: Optional[str] = None,
    min_stars: int = 100,
    max_repos: int = 50,
    shallow: bool = True,
    max_size_bytes: Optional[int] = None,
) -> tuple:
    """
    Download repositories for a specific category.
    
    Returns:
        (cloned_count, failed_count)
    """
    if category not in REPO_CATEGORIES:
        print(f"❌ Unknown category: {category}")
        return 0, 0
    
    cat_info = REPO_CATEGORIES[category]
    print(f"\n📦 Downloading {category} repositories...")
    print(f"   {cat_info['description']}")
    
    # For 'all-open' category, don't filter by license unless explicitly specified
    search_license = None if category == 'all-open' and not license else (license or 'mit')
    
    repos = search_github_repos(
        language=cat_info['language'],
        license=search_license,
        query=cat_info['query'],
        min_stars=min_stars,
        max_repos=max_repos,
    )
    
    if not repos:
        print(f"   No repositories found for {category}")
        return 0, 0
    
    print(f"   Cloning {len(repos)} repositories...")
    
    cloned = 0
    failed = 0
    
    # Progress bar for cloning
    pbar = tqdm(
        total=len(repos),
        desc=f"Cloning {category}",
        unit="repo",
        ncols=100,
        mininterval=0.1,
        maxinterval=1.0,
        file=sys.stderr,  # Write to stderr to avoid buffering issues
        dynamic_ncols=True,  # Auto-adjust to terminal width
        disable=False,  # Explicitly enable
    )
    
    # Cache size to avoid recalculating every iteration
    cached_size = get_directory_size(output_dir) if max_size_bytes else 0
    size_check_counter = 0
    
    for i, repo in enumerate(repos, 1):
        # Check size limit every 5 repos (to avoid blocking progress bar)
        if max_size_bytes:
            size_check_counter += 1
            if size_check_counter >= 5:
                cached_size = get_directory_size(output_dir)
                size_check_counter = 0
                if cached_size >= max_size_bytes:
                    pbar.close()
                    print(f"\n⚠️  Size limit reached: {format_size(cached_size)} >= {format_size(max_size_bytes)}")
                    print(f"   Stopping downloads for {category}.")
                    break
        
        repo_url = repo['clone_url']
        repo_name = repo['full_name']
        stars = repo['stargazers_count']
        repo_lang = repo.get('language', 'N/A')
        
        # Update progress bar before clone
        pbar.set_postfix({
            'Current': repo_name.split('/')[-1][:20],
            'Stars': f"{stars:,}",
            'Lang': repo_lang[:8],
            'Cloned': cloned,
            'Failed': failed,
            'Size': format_size(cached_size) if max_size_bytes else 'N/A'
        })
        
        success = clone_repo(
            repo_url,
            output_dir,
            depth=1 if shallow else None
        )
        
        if success:
            cloned += 1
        else:
            failed += 1
        
        # Update progress bar after clone (advance by 1)
        pbar.update(1)
        pbar.refresh()  # Force immediate refresh
        sys.stderr.flush()  # Force flush stderr to ensure progress bar displays
        
        # Rate limiting: small delay between clones
        time.sleep(0.5)
    
    pbar.close()
    
    return cloned, failed


def download_repos(
    output_dir: str = "data/repos",
    language: Optional[str] = None,
    license: Optional[str] = None,
    min_stars: int = 100,
    max_repos: int = 50,
    shallow: bool = True,
    languages: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    max_size_gb: Optional[float] = None,
) -> bool:
    """
    Download repositories from GitHub.
    
    Args:
        output_dir: Directory to clone repositories into
        language: Single language to filter by
        license: License type to filter by
        min_stars: Minimum stars
        max_repos: Maximum repos to download per category/language
        shallow: Use shallow clone (faster, less history)
        languages: List of languages to download
        categories: List of categories to download (nvim, lua, bash, zsh, python, hacking, security, all-open)
        max_size_gb: Maximum total size in GB (stops downloading when reached)
    
    Returns:
        True if successful
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert GB to bytes
    max_size_bytes = int(max_size_gb * 1024**3) if max_size_gb else None
    
    if max_size_bytes:
        current_size = get_directory_size(output_path)
        print(f"📊 Current directory size: {format_size(current_size)}")
        if current_size >= max_size_bytes:
            print(f"⚠️  Already at size limit: {format_size(current_size)} >= {format_size(max_size_bytes)}")
            return False
        print(f"📊 Size limit: {format_size(max_size_bytes)}")
    
    total_cloned = 0
    total_failed = 0
    
    # Download by categories
    if categories:
        print(f"\n📦 Processing {len(categories)} categories...")
        
        # Overall progress bar for categories
        cat_pbar = tqdm(
            categories,
            desc="Categories",
            unit="category",
            ncols=100,
            position=0,
            leave=True,
            mininterval=0.1,
            maxinterval=1.0,
            file=sys.stderr,  # Write to stderr to avoid buffering issues
            dynamic_ncols=True,  # Auto-adjust to terminal width
            disable=False,  # Explicitly enable
        )
        
        for category in cat_pbar:
            # Check size limit before processing category
            if max_size_bytes:
                current_size = get_directory_size(output_path)
                if current_size >= max_size_bytes:
                    cat_pbar.close()
                    print(f"\n⚠️  Size limit reached: {format_size(current_size)} >= {format_size(max_size_bytes)}")
                    print(f"   Stopping all downloads.")
                    break
            
            cat_pbar.set_description(f"Category: {category}")
            current_size = get_directory_size(output_path) if max_size_bytes else 0
            cat_pbar.set_postfix({
                'Total Cloned': total_cloned,
                'Total Failed': total_failed,
                'Size': format_size(current_size) if max_size_bytes else 'N/A'
            })
            cat_pbar.refresh()  # Force refresh
            
            cloned, failed = download_category(
                category=category,
                output_dir=output_path,
                license=license,
                min_stars=min_stars,
                max_repos=max_repos,
                shallow=shallow,
                max_size_bytes=max_size_bytes,
            )
            total_cloned += cloned
            total_failed += failed
        
        cat_pbar.close()
    
    # Download by languages
    languages_to_process = languages or ([language] if language else [])
    
    for lang in languages_to_process:
        # Check size limit
        if max_size_bytes:
            current_size = get_directory_size(output_path)
            if current_size >= max_size_bytes:
                print(f"\n⚠️  Size limit reached: {format_size(current_size)} >= {format_size(max_size_bytes)}")
                break
        
        print(f"\n📦 Processing {lang} repositories...")
        
        repos = search_github_repos(
            language=lang,
            license=license or 'mit',
            min_stars=min_stars,
            max_repos=max_repos,
        )
        
        if not repos:
            print(f"   No repositories found for {lang}")
            continue
        
        print(f"   Cloning {len(repos)} repositories...")
        
        # Progress bar for language-based cloning
        pbar = tqdm(
            total=len(repos),
            desc=f"Cloning {lang}",
            unit="repo",
            ncols=100,
            mininterval=0.1,
            maxinterval=1.0,
            file=sys.stderr,  # Write to stderr to avoid buffering issues
            dynamic_ncols=True,  # Auto-adjust to terminal width
            disable=False,  # Explicitly enable
        )
        
        # Cache size to avoid recalculating every iteration
        cached_size = get_directory_size(output_path) if max_size_bytes else 0
        size_check_counter = 0
        
        for i, repo in enumerate(repos, 1):
            # Check size limit every 5 repos
            if max_size_bytes:
                size_check_counter += 1
                if size_check_counter >= 5:
                    cached_size = get_directory_size(output_path)
                    size_check_counter = 0
                    if cached_size >= max_size_bytes:
                        pbar.close()
                        print(f"\n⚠️  Size limit reached: {format_size(cached_size)} >= {format_size(max_size_bytes)}")
                        break
            
            repo_url = repo['clone_url']
            repo_name = repo['full_name']
            stars = repo['stargazers_count']
            
            # Update progress bar before clone
            pbar.set_postfix({
                'Current': repo_name.split('/')[-1][:20],
                'Stars': f"{stars:,}",
                'Cloned': total_cloned,
                'Failed': total_failed,
                'Size': format_size(cached_size) if max_size_bytes else 'N/A'
            })
            
            success = clone_repo(
                repo_url,
                output_path,
                depth=1 if shallow else None
            )
            
            if success:
                total_cloned += 1
            else:
                total_failed += 1
            
            # Update progress bar after clone (advance by 1)
            pbar.update(1)
            pbar.refresh()  # Force immediate refresh
            sys.stderr.flush()  # Force flush stderr to ensure progress bar displays
            
            # Rate limiting
            time.sleep(0.5)
        
        pbar.close()
    
    final_size = get_directory_size(output_path) if max_size_bytes else 0
    print(f"\n✅ Download complete!")
    print(f"   Cloned: {total_cloned}")
    print(f"   Failed: {total_failed}")
    if max_size_bytes:
        print(f"   Total size: {format_size(final_size)} / {format_size(max_size_bytes)}")
    print(f"   Location: {output_path.absolute()}")
    
    return total_cloned > 0


def main():
    parser = argparse.ArgumentParser(
        description='Download GitHub repositories with open licenses for code training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Neovim configs
  python3 download_repos.py --categories nvim --max-repos 100
  
  # Download Lua repos
  python3 download_repos.py --categories lua --max-repos 50
  
  # Download Bash scripts
  python3 download_repos.py --categories bash --max-repos 50
  
  # Download ethical hacking repos
  python3 download_repos.py --categories hacking --max-repos 100
  
  # Download all your categories
  python3 download_repos.py --categories nvim lua bash zsh python hacking --max-repos 50
  
  # Download with 1 TB size limit
  python3 download_repos.py --categories all-open --max-repos 1000 --max-size 1024.0
  
  # Download with specific license
  python3 download_repos.py --categories nvim --license apache-2.0 --max-repos 50
        """
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/repos',
        help='Output directory (default: data/repos)'
    )
    parser.add_argument(
        '--language',
        type=str,
        choices=POPULAR_LANGUAGES,
        help='Programming language to filter by'
    )
    parser.add_argument(
        '--languages',
        type=str,
        nargs='+',
        choices=POPULAR_LANGUAGES,
        help='Multiple languages to download'
    )
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        choices=list(REPO_CATEGORIES.keys()),
        help='Categories to download: nvim, lua, bash, zsh, python, hacking, security, all-open'
    )
    parser.add_argument(
        '--license',
        type=str,
        choices=OPEN_LICENSES,
        default='mit',
        help='License type (default: mit)'
    )
    parser.add_argument(
        '--min-stars',
        type=int,
        default=100,
        help='Minimum stars (default: 100)'
    )
    parser.add_argument(
        '--max-repos',
        type=int,
        default=50,
        help='Maximum repos per category/language (default: 50)'
    )
    parser.add_argument(
        '--full-clone',
        action='store_true',
        help='Do full clone instead of shallow (slower but includes full history)'
    )
    parser.add_argument(
        '--max-size',
        type=float,
        help='Maximum total size in GB (stops downloading when reached, e.g., 1024.0 for 1 TB)'
    )
    
    args = parser.parse_args()
    
    # Default to categories if nothing specified
    if not args.categories and not args.language and not args.languages:
        print("ℹ️  No categories or languages specified. Use --categories or --language")
        print("   Available categories:", ", ".join(REPO_CATEGORIES.keys()))
        print("   Example: --categories nvim lua bash hacking")
        return
    
    print("🚀 SheepOp Repository Downloader")
    print("=" * 50)
    
    success = download_repos(
        output_dir=args.output,
        language=args.language,
        license=args.license,
        min_stars=args.min_stars,
        max_repos=args.max_repos,
        shallow=not args.full_clone,
        languages=args.languages,
        categories=args.categories,
        max_size_gb=args.max_size,
    )
    
    if success:
        print(f"\n📚 You can now train with:")
        print(f"   python3 train.py --data {args.output} --config config.json --device cuda")
    else:
        print("\n❌ No repositories were downloaded.")
        sys.exit(1)


if __name__ == '__main__':
    main()

