#!/usr/bin/env python3
"""
Migration Script: HeavyDB to Parquet/Arrow/cuDF
Helps transition existing workflows to new architecture
"""

import os
import sys
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParquetMigrator:
    """Handles migration from HeavyDB to Parquet/Arrow/cuDF"""
    
    def __init__(self, backup_dir: str = None):
        """Initialize migrator"""
        self.backup_dir = backup_dir or f"/mnt/optimizer_share/migration_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.migration_log = []
        
    def backup_existing_files(self):
        """Backup existing workflow files"""
        logger.info("Backing up existing files...")
        
        files_to_backup = [
            "/mnt/optimizer_share/backend/csv_only_heavydb_workflow.py",
            "/mnt/optimizer_share/backend/samba_job_queue_processor.py",
            "/mnt/optimizer_share/config/production_config.ini"
        ]
        
        os.makedirs(self.backup_dir, exist_ok=True)
        
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                backup_path = os.path.join(self.backup_dir, os.path.basename(file_path))
                shutil.copy2(file_path, backup_path)
                logger.info(f"Backed up: {file_path} -> {backup_path}")
                self.migration_log.append(f"Backed up {file_path}")
    
    def update_job_processor(self):
        """Update job queue processor to use Parquet workflow"""
        logger.info("Updating job queue processor...")
        
        processor_path = "/mnt/optimizer_share/backend/samba_job_queue_processor.py"
        if not os.path.exists(processor_path):
            logger.warning(f"Job processor not found: {processor_path}")
            return
        
        # Read existing processor
        with open(processor_path, 'r') as f:
            content = f.read()
        
        # Update imports and workflow references
        updates = [
            ("from csv_only_heavydb_workflow import", "from parquet_cudf_workflow import"),
            ("csv_only_heavydb_workflow.py", "parquet_cudf_workflow.py"),
            ("HeavyDB workflow", "Parquet/cuDF workflow")
        ]
        
        for old, new in updates:
            content = content.replace(old, new)
        
        # Write updated processor
        with open(processor_path, 'w') as f:
            f.write(content)
        
        logger.info("Updated job queue processor")
        self.migration_log.append("Updated job queue processor to use Parquet workflow")
    
    def update_configuration(self):
        """Update configuration files"""
        logger.info("Updating configuration...")
        
        # Add Parquet configuration to production config
        config_updates = {
            'data_format': 'parquet',
            'use_gpu': 'true',
            'parquet_compression': 'snappy',
            'arrow_memory_pool_gb': '4.0',
            'cudf_enabled': 'true'
        }
        
        config_path = "/mnt/optimizer_share/config/production_config.ini"
        if os.path.exists(config_path):
            # Read existing config
            with open(config_path, 'r') as f:
                lines = f.readlines()
            
            # Add new section if not exists
            if '[parquet]' not in ''.join(lines):
                lines.append('\n[parquet]\n')
                for key, value in config_updates.items():
                    lines.append(f'{key} = {value}\n')
                
                # Write updated config
                with open(config_path, 'w') as f:
                    f.writelines(lines)
                
                logger.info("Added Parquet configuration section")
                self.migration_log.append("Added Parquet configuration to production config")
    
    def convert_existing_data(self, data_dir: str = "/mnt/optimizer_share/input"):
        """Convert existing CSV files to Parquet"""
        logger.info("Converting existing CSV files to Parquet...")
        
        from lib.parquet_pipeline import csv_to_parquet
        
        parquet_dir = Path("/mnt/optimizer_share/data/parquet/strategies")
        parquet_dir.mkdir(parents=True, exist_ok=True)
        
        csv_files = list(Path(data_dir).glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to convert")
        
        converted = 0
        for csv_file in csv_files:
            try:
                parquet_file = parquet_dir / f"{csv_file.stem}.parquet"
                
                # Skip if already converted
                if parquet_file.exists():
                    logger.info(f"Skipping {csv_file.name} - already converted")
                    continue
                
                # Convert to Parquet
                success = csv_to_parquet(str(csv_file), str(parquet_file))
                
                if success:
                    converted += 1
                    logger.info(f"Converted: {csv_file.name} -> {parquet_file.name}")
                else:
                    logger.warning(f"Failed to convert: {csv_file.name}")
                    
            except Exception as e:
                logger.error(f"Error converting {csv_file.name}: {str(e)}")
        
        logger.info(f"Converted {converted}/{len(csv_files)} CSV files")
        self.migration_log.append(f"Converted {converted} CSV files to Parquet format")
    
    def create_rollback_script(self):
        """Create script to rollback migration if needed"""
        rollback_script = f"""#!/bin/bash
# Rollback script for Parquet migration
# Generated on {datetime.now().isoformat()}

echo "Rolling back Parquet migration..."

# Restore backed up files
cp {self.backup_dir}/csv_only_heavydb_workflow.py /mnt/optimizer_share/backend/
cp {self.backup_dir}/samba_job_queue_processor.py /mnt/optimizer_share/backend/
cp {self.backup_dir}/production_config.ini /mnt/optimizer_share/config/

echo "Rollback completed. Original HeavyDB workflow restored."
"""
        
        rollback_path = os.path.join(self.backup_dir, "rollback.sh")
        with open(rollback_path, 'w') as f:
            f.write(rollback_script)
        
        os.chmod(rollback_path, 0o755)
        logger.info(f"Created rollback script: {rollback_path}")
        self.migration_log.append(f"Created rollback script at {rollback_path}")
    
    def verify_migration(self):
        """Verify migration was successful"""
        logger.info("Verifying migration...")
        
        checks = {
            'Parquet workflow exists': os.path.exists("/mnt/optimizer_share/backend/parquet_cudf_workflow.py"),
            'Parquet pipeline module': os.path.exists("/mnt/optimizer_share/backend/lib/parquet_pipeline/__init__.py"),
            'Arrow connector module': os.path.exists("/mnt/optimizer_share/backend/lib/arrow_connector/__init__.py"),
            'cuDF engine module': os.path.exists("/mnt/optimizer_share/backend/lib/cudf_engine/__init__.py"),
            'Parquet data directory': os.path.exists("/mnt/optimizer_share/data/parquet/strategies"),
            'Configuration updated': os.path.exists("/mnt/optimizer_share/config/parquet_arrow_config.ini")
        }
        
        all_passed = True
        for check, result in checks.items():
            status = "✅" if result else "❌"
            logger.info(f"{status} {check}")
            if not result:
                all_passed = False
        
        return all_passed
    
    def save_migration_report(self):
        """Save migration report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'backup_directory': self.backup_dir,
            'migration_steps': self.migration_log,
            'status': 'completed'
        }
        
        report_path = os.path.join(self.backup_dir, "migration_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Migration report saved to: {report_path}")
    
    def run_migration(self, skip_data_conversion: bool = False):
        """Run complete migration process"""
        logger.info("=" * 60)
        logger.info("Starting HeavyDB to Parquet/Arrow/cuDF Migration")
        logger.info("=" * 60)
        
        try:
            # Step 1: Backup
            self.backup_existing_files()
            
            # Step 2: Update job processor
            self.update_job_processor()
            
            # Step 3: Update configuration
            self.update_configuration()
            
            # Step 4: Convert data (optional)
            if not skip_data_conversion:
                self.convert_existing_data()
            else:
                logger.info("Skipping data conversion as requested")
            
            # Step 5: Create rollback script
            self.create_rollback_script()
            
            # Step 6: Verify migration
            success = self.verify_migration()
            
            # Step 7: Save report
            self.save_migration_report()
            
            if success:
                logger.info("\n✅ Migration completed successfully!")
                logger.info(f"Backup saved to: {self.backup_dir}")
                logger.info(f"To rollback, run: {self.backup_dir}/rollback.sh")
            else:
                logger.error("\n❌ Migration verification failed!")
                logger.error("Some components are missing. Check the logs above.")
            
            return success
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Migrate from HeavyDB to Parquet/Arrow/cuDF')
    parser.add_argument('--backup-dir', help='Directory for backup files')
    parser.add_argument('--skip-data', action='store_true', help='Skip CSV to Parquet conversion')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("\nMigration would perform the following steps:")
        logger.info("1. Backup existing workflow files")
        logger.info("2. Update job queue processor to use Parquet workflow")
        logger.info("3. Update configuration files")
        logger.info("4. Convert existing CSV files to Parquet format")
        logger.info("5. Create rollback script")
        logger.info("6. Verify migration")
        logger.info("7. Save migration report")
        return
    
    # Create migrator
    migrator = ParquetMigrator(backup_dir=args.backup_dir)
    
    # Run migration
    success = migrator.run_migration(skip_data_conversion=args.skip_data)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()