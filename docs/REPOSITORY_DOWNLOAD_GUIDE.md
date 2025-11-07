# Repository Download Guide

This guide explains how to automatically download GitHub repositories with open licenses for code training using the repository downloader scripts.

## Overview

The repository downloader allows you to automatically find and clone GitHub repositories based on:
- **Categories**: Neovim configs, Lua repos, Bash scripts, Zsh configs, Python repos, ethical hacking tools, security tools, and all open-license repos
- **Languages**: Python, JavaScript, Go, Rust, and 15+ more
- **Licenses**: MIT, Apache, BSD, GPL, and other open source licenses
- **Quality**: Filter by minimum stars (popularity)
- **Size Limits**: Automatic stopping when reaching storage limits (default: 1 TB)

## Scripts

There are two scripts available:

1. **`download_all_repos.py`** - Convenience script to download all common categories at once
2. **`download_repos.py`** - Full-featured script with all options and flexibility

## Quick Start

### Download All Categories (Recommended)

The easiest way to download all repository categories:

```bash
python3 download_all_repos.py
```

This will download:
- 📦 Neovim configurations and plugins
- 📦 Lua programming repositories
- 📦 Bash/shell script repositories
- 📦 Zsh configuration and plugins
- 📦 Python programming repositories
- 📦 Ethical hacking and cybersecurity tools

**Default settings:**
- Max repos per category: 50
- Min stars: 100
- Output directory: `data/repos`
- Size limit: 1 TB (1024 GB)
- Shallow clones (faster, less disk space)

### Download Specific Categories

```bash
python3 download_repos.py --categories nvim lua bash zsh python hacking --max-repos 50
```

### Download All Open-License Repos

Download repositories with any open license (any language):

```bash
python3 download_repos.py --categories all-open --max-repos 1000 --max-size 1024.0
```

### Download by Language

```bash
python3 download_repos.py --language python --max-repos 100
```

## Installation

No additional dependencies required! The script uses:
- Python standard library (`urllib`, `json`, `subprocess`)
- `tqdm` (already in requirements.txt)
- `git` (should be installed on your system)

## Available Categories

### Neovim (`nvim`)
Neovim configuration files and plugins written in Lua.

```bash
python3 download_repos.py --categories nvim --max-repos 100
```

**What it searches for:**
- `neovim OR nvim-config OR neovim-config`
- MIT licensed repositories (default)
- 100+ stars minimum (default)

### Lua (`lua`)
Lua programming language repositories.

```bash
python3 download_repos.py --categories lua --max-repos 50
```

**What it searches for:**
- Language: Lua
- MIT licensed repositories (default)
- 100+ stars minimum (default)

### Bash (`bash`)
Bash and shell script repositories.

```bash
python3 download_repos.py --categories bash --max-repos 50
```

**What it searches for:**
- Language: Shell
- MIT licensed repositories (default)
- 100+ stars minimum (default)

### Zsh (`zsh`)
Zsh configuration files and plugins (Oh My Zsh, etc.).

```bash
python3 download_repos.py --categories zsh --max-repos 50
```

**What it searches for:**
- `zsh-config OR oh-my-zsh OR zsh-plugin`
- MIT licensed repositories (default)
- 100+ stars minimum (default)

### Python (`python`)
Python programming language repositories.

```bash
python3 download_repos.py --categories python --max-repos 100
```

**What it searches for:**
- Language: Python
- MIT licensed repositories (default)
- 100+ stars minimum (default)

### Ethical Hacking (`hacking`)
Ethical hacking and cybersecurity tools.

```bash
python3 download_repos.py --categories hacking --max-repos 100
```

**What it searches for:**
- `ethical-hacking OR cybersecurity OR penetration-testing OR security-tools OR red-team`
- MIT licensed repositories (default)
- 100+ stars minimum (default)

### Security (`security`)
General security and cybersecurity repositories.

```bash
python3 download_repos.py --categories security --max-repos 50
```

**What it searches for:**
- `security-tools OR cybersecurity OR penetration-testing OR red-team OR blue-team`
- MIT licensed repositories (default)
- 100+ stars minimum (default)

### All Open Licenses (`all-open`)
All repositories with open licenses, any language. This is useful for downloading a diverse set of repositories.

```bash
python3 download_repos.py --categories all-open --max-repos 1000 --max-size 1024.0
```

**What it searches for:**
- Any open-license repository (no language filter)
- No specific license filter (searches all open licenses)
- 100+ stars minimum (default)

**Note:** This category searches broadly and may return repositories with various licenses. You can still specify `--license` to filter to a specific license type.

## Command-Line Options

### `download_repos.py` Options

```bash
python3 download_repos.py [OPTIONS]
```

**Options:**

- `--output DIR` - Output directory (default: `data/repos`)
- `--categories CAT1 CAT2 ...` - Categories to download: `nvim`, `lua`, `bash`, `zsh`, `python`, `hacking`, `security`, `all-open`
- `--language LANG` - Single language to filter by
- `--languages LANG1 LANG2 ...` - Multiple languages to download
- `--license LICENSE` - License type (default: `mit`)
- `--min-stars N` - Minimum stars (default: 100)
- `--max-repos N` - Maximum repos per category/language (default: 50)
- `--max-size N` - Maximum total size in GB (stops downloading when reached, e.g., `1024.0` for 1 TB)
- `--full-clone` - Do full clone instead of shallow (slower but includes full history)

### `download_all_repos.py` Options

```bash
python3 download_all_repos.py [OPTIONS]
```

**Options:**

- `--max-repos N` - Maximum repos per category (default: 50)
- `--min-stars N` - Minimum stars (default: 100)
- `--output DIR` - Output directory (default: `data/repos`)
- `--max-size N` - Maximum total size in GB (default: 1024.0 = 1 TB)
- `--full-clone` - Do full clone instead of shallow

**Example:**
```bash
python3 download_all_repos.py --max-repos 100 --min-stars 200 --max-size 2048.0
```

### Available Licenses

- `mit` (default)
- `apache-2.0`
- `bsd-3-clause`
- `bsd-2-clause`
- `isc`
- `unlicense`
- `mpl-2.0`
- `lgpl-2.1`
- `lgpl-3.0`
- `gpl-2.0`
- `gpl-3.0`

### Available Languages

- `python`
- `javascript`
- `typescript`
- `java`
- `cpp`
- `c`
- `go`
- `rust`
- `ruby`
- `php`
- `swift`
- `kotlin`
- `scala`
- `r`
- `sql`
- `lua`
- `shell` (for bash/shell scripts)

## Examples

### Example 1: Download All Categories (Simple)

```bash
python3 download_all_repos.py
```

Downloads all categories (nvim, lua, bash, zsh, python, hacking) with default settings and 1 TB size limit.

### Example 2: Download All Categories with Custom Settings

```bash
python3 download_all_repos.py --max-repos 100 --min-stars 200 --max-size 2048.0
```

Downloads all categories with:
- 100 repos per category
- Minimum 200 stars
- 2 TB size limit

### Example 3: Download Specific Categories

```bash
python3 download_repos.py --categories nvim lua bash zsh python hacking --max-repos 50
```

Downloads specific categories with 50 repos each.

### Example 4: Download All Open-License Repos with Size Limit

```bash
python3 download_repos.py --categories all-open --max-repos 1000 --max-size 1024.0
```

Downloads up to 1000 repositories with any open license, stopping at 1 TB.

### Example 5: Download High-Quality Repos

```bash
python3 download_repos.py --categories nvim lua bash zsh python hacking --min-stars 1000 --max-repos 20
```

Downloads only highly popular repositories (1000+ stars).

### Example 6: Download Multiple Languages

```bash
python3 download_repos.py --languages python javascript go rust --max-repos 50
```

Downloads repositories in multiple programming languages.

### Example 7: Download with Apache License

```bash
python3 download_repos.py --categories nvim --license apache-2.0 --max-repos 50
```

Downloads Neovim repos with Apache 2.0 license.

### Example 8: Custom Output Directory

```bash
python3 download_repos.py --categories nvim lua bash zsh python hacking --output /path/to/repos
```

Saves repositories to a custom directory.

### Example 9: Full Clone (with History)

```bash
python3 download_repos.py --categories nvim --full-clone --max-repos 10
```

Does full clone including full git history (slower but more complete).

### Example 10: Size-Limited Download

```bash
python3 download_repos.py --categories all-open --max-repos 2000 --max-size 512.0
```

Downloads repositories but stops when reaching 512 GB (0.5 TB).

## Progress Tracking

The scripts include visual progress bars showing:

- **Category progress**: Overall progress across all categories
- **Repository progress**: Progress for each category
- **Real-time statistics**: Current repo, stars, language, cloned/failed counts
- **Size tracking**: Current total size and size limit (when `--max-size` is used)

**Example output:**

```text
📊 Current directory size: 45.23 GB
📊 Size limit: 1024.00 GB
📦 Processing 6 categories...
Category: nvim: 100%|████████████| 6/6 [15:23<00:00, Size=156.78 GB, Total Cloned=300, Total Failed=2]
Cloning nvim: 45%|████████████████▌                    | 23/50 [02:15<03:45, Current=awesome-nvim, Stars=5.2k, Lang=Lua, Cloned=22, Failed=1, Size=12.45 GB]
```

**Size limit reached:**

When the size limit is reached, the script will stop downloading and show:

```text
⚠️  Size limit reached: 1024.00 GB >= 1024.00 GB
   Stopping all downloads.
```

## GitHub API Rate Limits

GitHub API has rate limits:
- **Unauthenticated**: 60 requests/hour
- **Authenticated**: 5,000 requests/hour

### Using a GitHub Token

To increase rate limits, set a GitHub Personal Access Token:

```bash
export GITHUB_TOKEN=your_token_here
python3 download_repos.py --categories nvim lua bash hacking
```

**How to create a token:**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scope: `public_repo` (read-only is enough)
4. Copy token and set as environment variable

## Size Limits

The repository downloader includes automatic size limit checking to prevent running out of disk space.

### How It Works

- **Default limit**: 1 TB (1024 GB) for `download_all_repos.py`
- **Customizable**: Use `--max-size` to set any limit
- **Real-time tracking**: Size is checked before each repository clone
- **Automatic stopping**: Downloads stop when limit is reached
- **Progress display**: Current size shown in progress bars

### Setting Size Limits

**With `download_all_repos.py`:**
```bash
# Default 1 TB
python3 download_all_repos.py

# Custom limit (2 TB)
python3 download_all_repos.py --max-size 2048.0

# Smaller limit (500 GB)
python3 download_all_repos.py --max-size 512.0
```

**With `download_repos.py`:**
```bash
# No limit (downloads until max-repos reached)
python3 download_repos.py --categories nvim --max-repos 100

# With 1 TB limit
python3 download_repos.py --categories nvim --max-repos 1000 --max-size 1024.0
```

### Size Calculation

The script calculates total size by:
- Scanning all files in the output directory (`data/repos` by default)
- Summing file sizes recursively
- Checking before each new repository clone
- Displaying human-readable sizes (B, KB, MB, GB, TB)

**Note:** Size checking happens before cloning, so the actual size may be slightly less than the limit when stopping.

## Cache and Resuming

The scripts automatically:

- **Skips existing repos**: If a repository already exists, it's skipped (no re-download)
- **Resumes downloads**: You can run the script multiple times safely
- **Progress tracking**: Shows what's already downloaded
- **Size awareness**: Accounts for existing repositories when checking size limits

After downloading repositories, they're automatically processed during training:

```bash
# Download repos
python3 download_all_repos.py

# Train with all data (text + code)
python3 train.py --data data/ --config config.json --device cuda
```

The training script will:
1. Process all your text data (Wiki, Books, Amazon reviews, etc.)
2. Process all code repositories
3. Combine everything into training data

## Supported File Types

The data processor automatically handles code files from repositories:

- **Text files**: `.txt`, `.md`, `.rst`, `.log`, `.csv`, `.json`, `.jsonl`, `.xml`, `.html`, `.htm`
- **Code files**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, `.lua`, `.sh`, and 30+ more
- **PDF files**: `.pdf` (if pdfplumber is installed)
- **Images**: `.png`, `.jpg`, etc. (if OCR is set up)

## Troubleshooting

### Rate Limit Exceeded

**Error:** `Rate limit exceeded`

**Solution:**
1. Wait a few minutes and try again
2. Use a GitHub token: `export GITHUB_TOKEN=your_token`
3. Reduce `--max-repos` to download fewer repos per run

### Repository Clone Fails

**Error:** `Failed to clone repository`

**Possible causes:**
- Repository was deleted or made private
- Network issues
- Repository is too large (timeout)

**Solution:**
- The script continues with other repos
- Failed repos are counted and reported at the end
- You can re-run the script to retry failed repos

### No Repositories Found

**Error:** `No repositories found`

**Possible causes:**
- Search query too restrictive
- License filter too narrow
- Minimum stars too high

**Solution:**
- Lower `--min-stars` threshold
- Try different `--license` options
- Check if category name is correct

## Best Practices

### 1. Start Small

Test with a small number first:

```bash
python3 download_repos.py --categories nvim --max-repos 10
```

### 2. Use Size Limits

Always set a size limit to prevent running out of disk space:

```bash
# Recommended: 1 TB limit
python3 download_all_repos.py --max-size 1024.0

# Or custom limit based on available space
python3 download_repos.py --categories all-open --max-size 512.0
```

### 3. Use Shallow Clones

Shallow clones are faster and use less disk space:

```bash
# Default (shallow clone)
python3 download_repos.py --categories nvim

# Full clone (only if you need history)
python3 download_repos.py --categories nvim --full-clone
```

### 4. Filter by Quality

Use `--min-stars` to get quality repositories:

```bash
python3 download_repos.py --categories nvim --min-stars 500 --max-repos 50
```

### 5. Use GitHub Token

For large downloads, use a GitHub token:

```bash
export GITHUB_TOKEN=your_token_here
python3 download_all_repos.py --max-repos 100
```

### 6. Monitor Disk Space

Check available disk space before starting:

```bash
df -h data/repos
```

### 7. Use `all-open` Category Wisely

The `all-open` category downloads broadly. Consider:
- Setting a reasonable `--max-repos` limit
- Using `--min-stars` to filter quality
- Setting `--max-size` to prevent excessive downloads

```bash
python3 download_repos.py --categories all-open --max-repos 500 --min-stars 200 --max-size 1024.0
```

## Storage Considerations

### Size Limits and Disk Space Management

- **Default**: 1 TB (1024 GB) for `download_all_repos.py`
- **Recommended**: Set based on available disk space
- **Monitoring**: Script shows current size vs limit in progress bars

### Shallow vs Full Clones

**Shallow clones (default):**
- Faster download
- Less disk space (~10-50% of full clone)
- No git history
- Good for training data

**Full clones:**
- Slower download
- More disk space (includes full history)
- Includes full git history
- Useful if you need version history

**Typical sizes (shallow clones):**
- Small repo: 1-10 MB
- Medium repo: 10-100 MB
- Large repo: 100 MB - 1 GB
- Very large repo: 1-10 GB+

**Example:** Downloading 300 repositories with shallow clones typically uses 5-30 GB, depending on repository sizes.

### Estimating Storage Needs

To estimate how many repositories you can download:

1. **Check current size:**
   ```bash
   du -sh data/repos
   ```

2. **Calculate average repo size:**
   - Small repos: ~5 MB average
   - Medium repos: ~50 MB average
   - Large repos: ~500 MB average

3. **Estimate:**
   - 100 small repos: ~500 MB
   - 100 medium repos: ~5 GB
   - 100 large repos: ~50 GB
   - 1000 mixed repos: ~50-200 GB

4. **Set appropriate limit:**
   ```bash
   # For 1 TB available space, use 900 GB limit (leave buffer)
   python3 download_all_repos.py --max-size 900.0
   ```

## Summary

The repository downloader makes it easy to:
- ✅ Automatically find high-quality open-source repositories
- ✅ Filter by category, language, license, and popularity
- ✅ Download with progress tracking and size monitoring
- ✅ Set size limits to prevent running out of disk space
- ✅ Integrate seamlessly with training pipeline
- ✅ Resume interrupted downloads

**Available categories:**
- `nvim` - Neovim configurations and plugins
- `lua` - Lua programming repositories
- `bash` - Bash/shell script repositories
- `zsh` - Zsh configuration and plugins
- `python` - Python programming repositories
- `hacking` - Ethical hacking and cybersecurity tools
- `security` - Security and cybersecurity repositories
- `all-open` - All repositories with open licenses (any language)

**Quick commands to get started:**

```bash
# Download all categories with 1 TB limit (recommended)
python3 download_all_repos.py

# Download specific categories
python3 download_repos.py --categories nvim lua bash zsh python hacking --max-repos 50

# Download all open-license repos with size limit
python3 download_repos.py --categories all-open --max-repos 1000 --max-size 1024.0
```

This downloads repositories and prepares them for training!