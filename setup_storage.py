#!/usr/bin/env python3
"""
Move large files to external storage and create symbolic links.
Helps manage large datasets and checkpoints on systems with limited space.
"""
import os
import shutil
import subprocess
from pathlib import Path
import argparse


def create_storage_structure(storage_root: str):
    """Create directory structure in storage location."""
    storage_path = Path(storage_root)
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (storage_path / "data").mkdir(exist_ok=True)
    (storage_path / "checkpoints").mkdir(exist_ok=True)
    (storage_path / "checkpoints_test").mkdir(exist_ok=True)
    
    print(f"✅ Created storage structure at: {storage_path}")
    return storage_path


def move_and_link(source_dir: Path, target_dir: Path, link_name: str, dry_run: bool = False):
    """
    Move directory contents to storage and create symbolic link.
    
    Args:
        source_dir: Source directory in project
        target_dir: Target directory in storage
        link_name: Name for the symbolic link (same as source_dir name)
        dry_run: If True, only show what would be done
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    if not source_dir.exists():
        print(f"⚠️  Source directory doesn't exist: {source_dir}")
        return False
    
    if dry_run:
        print(f"\n[DRY RUN] Would move contents from {source_dir} to {target_dir}")
        print(f"         Would replace {source_dir} with symlink -> {target_dir}")
        return True
    
    # Move files (skip Python module files)
    moved_count = 0
    temp_backup = source_dir.parent / f".{source_dir.name}_backup"
    
    # First, backup Python files
    python_files = []
    for item in source_dir.iterdir():
        if item.suffix == '.py' or item.name.startswith('__'):
            python_files.append(item)
    
    # Move non-Python files to storage
    for item in source_dir.iterdir():
        # Skip Python files and hidden files
        if item.name.startswith('__') or item.suffix == '.py' or item.name.startswith('.'):
            continue
        
        target_item = target_dir / item.name
        if target_item.exists():
            print(f"⚠️  Skipping {item.name} (already exists in storage)")
            continue
        
        print(f"📦 Moving {item.name}...")
        try:
            if item.is_dir():
                shutil.copytree(item, target_item)
                shutil.rmtree(item)
            else:
                shutil.copy2(item, target_item)
                item.unlink()
            moved_count += 1
        except Exception as e:
            print(f"❌ Error moving {item.name}: {e}")
            return False
    
    # Copy Python files to storage (keep structure)
    for item in python_files:
        target_item = target_dir / item.name
        if not target_item.exists():
            shutil.copy2(item, target_item)
    
    # Replace source directory with symlink
    # Step 1: Remove original directory
    try:
        shutil.rmtree(source_dir)
    except Exception as e:
        print(f"⚠️  Could not remove {source_dir}: {e}")
        return False
    
    # Step 2: Create symlink
    try:
        source_dir.symlink_to(target_dir)
        print(f"✅ Created symbolic link: {source_dir} -> {target_dir}")
        return True
    except Exception as e:
        print(f"❌ Error creating link: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Move large files to external storage and create symbolic links',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (show what would be done)
  python3 setup_storage.py --storage /mnt/storage/sheepOp --dry-run
  
  # Move data and checkpoints to storage
  python3 setup_storage.py --storage /mnt/storage/sheepOp
  
  # Only move data, not checkpoints
  python3 setup_storage.py --storage /mnt/storage/sheepOp --skip-checkpoints
        """
    )
    
    parser.add_argument(
        '--storage',
        type=str,
        default='/mnt/storage/sheepOp',
        help='Storage root directory (default: /mnt/storage/sheepOp)'
    )
    
    parser.add_argument(
        '--project-root',
        type=str,
        default='.',
        help='Project root directory (default: current directory)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )
    
    parser.add_argument(
        '--skip-checkpoints',
        action='store_true',
        help='Skip moving checkpoints (only move data)'
    )
    
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip moving data (only move checkpoints)'
    )
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    storage_root = Path(args.storage).resolve()
    
    print(f"🚀 Setting up storage links")
    print(f"   Project root: {project_root}")
    print(f"   Storage root: {storage_root}")
    print(f"   Dry run: {args.dry_run}\n")
    
    # Create storage structure (always create, even in dry-run, to check permissions)
    try:
        create_storage_structure(args.storage)
    except Exception as e:
        if args.dry_run:
            print(f"⚠️  Could not create storage structure (will be created during actual run): {e}")
        else:
            print(f"❌ Error creating storage structure: {e}")
            return 1
    
    storage_data = storage_root / "data"
    storage_checkpoints = storage_root / "checkpoints"
    storage_checkpoints_test = storage_root / "checkpoints_test"
    
    project_data = project_root / "data"
    project_checkpoints = project_root / "checkpoints"
    project_checkpoints_test = project_root / "checkpoints_test"
    
    success = True
    
    # Move data
    if not args.skip_data:
        print(f"\n📁 Processing data directory...")
        if project_data.exists():
            if project_data.is_symlink():
                print(f"   ℹ️  data/ is already a symlink: {project_data.readlink()}")
            else:
                print("   Moving data files to storage (keeping __init__.py)...")
                # Copy __init__.py to storage first
                init_file = project_data / "__init__.py"
                if init_file.exists():
                    storage_init = storage_data / "__init__.py"
                    if not storage_init.exists():
                        if args.dry_run:
                            print(f"   [DRY RUN] Would copy: __init__.py -> {storage_init}")
                        else:
                            # Ensure storage directory exists
                            storage_data.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(init_file, storage_init)
                            print("   ✅ Copied __init__.py to storage")
                    else:
                        print("   ℹ️  __init__.py already exists in storage")
                
                # Move all other files
                moved_files = []
                for item in project_data.iterdir():
                    if item.name == '__init__.py' or item.name.startswith('__'):
                        continue
                    
                    target_item = storage_data / item.name
                    if args.dry_run:
                        print(f"   [DRY RUN] Would move: {item.name} -> {target_item}")
                        moved_files.append(item.name)
                    else:
                        if not target_item.exists():
                            if item.is_dir():
                                shutil.copytree(item, target_item)
                                shutil.rmtree(item)
                            else:
                                shutil.copy2(item, target_item)
                                item.unlink()
                            moved_files.append(item.name)
                            print(f"   ✅ Moved: {item.name}")
                        else:
                            print(f"   ⚠️  Already exists: {item.name}")
                
                # Replace data/ with symlink
                if not args.dry_run and moved_files:
                    # Remove original directory
                    init_backup = project_data / "__init__.py"
                    if init_backup.exists():
                        # Keep a reference
                        pass
                    shutil.rmtree(project_data)
                    
                    # Create symlink
                    project_data.symlink_to(storage_data)
                    print(f"   ✅ Replaced data/ with symlink -> {storage_data}")
        else:
            print("   ℹ️  data/ directory doesn't exist, creating symlink...")
            if not args.dry_run:
                project_data.symlink_to(storage_data)
                print(f"   ✅ Created data/ symlink -> {storage_data}")
    
    # Move checkpoints
    if not args.skip_checkpoints:
        print(f"\n💾 Processing checkpoints...")
        
        if project_checkpoints.exists():
            print("   Moving checkpoints to storage...")
            success = move_and_link(
                project_checkpoints,
                storage_checkpoints,
                "checkpoints",
                args.dry_run
            ) and success
        
        if project_checkpoints_test.exists():
            print("   Moving checkpoints_test to storage...")
            success = move_and_link(
                project_checkpoints_test,
                storage_checkpoints_test,
                "checkpoints_test",
                args.dry_run
            ) and success
    
    if args.dry_run:
        print(f"\n✅ Dry run complete. Use without --dry-run to execute.")
    else:
        if success:
            print(f"\n✅ Storage setup complete!")
            print(f"\n📋 Next steps:")
            print(f"   1. Your data files are now in: {storage_root}/data/")
            print(f"   2. Your checkpoints will be saved to: {storage_root}/checkpoints/")
            print(f"   3. Links are created in your project directory")
            print(f"   4. Training will automatically use the storage location")
        else:
            print(f"\n❌ Some operations failed. Please check the errors above.")
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

