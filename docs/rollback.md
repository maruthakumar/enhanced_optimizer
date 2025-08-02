# Rollback & Dual-Write Procedures

**Document Owner:** PO Agent  
**Version:** 1.0  
**Date:** August 2, 2025  
**Status:** Production Ready  

## Overview

This document provides comprehensive rollback and dual-write procedures for the HeavyDB → Parquet/Arrow/cuDF migration of the Heavy Optimizer Platform. It serves as an operational runbook for maintaining zero downtime during migration phases while ensuring data integrity and system reliability.

## Quick Reference Emergency Commands

### Immediate Rollback to HeavyDB
```bash
# Emergency rollback to stable HeavyDB system
cd /mnt/optimizer_share
./scripts/emergency_rollback.sh

# Manual rollback if script fails
sudo systemctl stop samba_job_queue_processor
sed -i 's/DATA_BACKEND=parquet/DATA_BACKEND=heavydb/g' /mnt/optimizer_share/config/production_config.ini
sudo systemctl start samba_job_queue_processor
```

### Check Migration Status
```bash
# View current backend configuration
grep "DATA_BACKEND\|backend.*=" /mnt/optimizer_share/config/production_config.ini

# Check active job processor
ps aux | grep -E "samba_job_queue_processor|csv_only_.*_workflow"

# Monitor job queue status
ls -la /mnt/optimizer_share/jobs/{queue,processing,failed}/
```

## Data Backend Configuration

### Backend Toggle Mechanism

The system supports runtime backend switching via configuration flags in `/mnt/optimizer_share/config/production_config.ini`:

```ini
[SYSTEM]
# Primary backend selection
DATA_BACKEND = heavydb          # Options: heavydb, parquet
migration_phase = dual_write    # Options: heavydb_only, dual_write, cut_over, parquet_only

[MIGRATION]
# Dual-write configuration
enable_dual_write = true
primary_backend = heavydb
secondary_backend = parquet
validation_enabled = true
rollback_on_validation_failure = true

# Cut-over configuration  
traffic_percentage_parquet = 0   # 0-100: Percentage of jobs routed to Parquet
```

### Backend Switch Commands

#### Switch to HeavyDB Only
```bash
cd /mnt/optimizer_share

# Update configuration
sed -i 's/DATA_BACKEND=.*/DATA_BACKEND=heavydb/g' config/production_config.ini
sed -i 's/migration_phase=.*/migration_phase=heavydb_only/g' config/production_config.ini
sed -i 's/enable_dual_write=.*/enable_dual_write=false/g' config/production_config.ini

# Restart job processor to apply changes
sudo systemctl restart samba_job_queue_processor

# Verify switch
tail -f logs/job_processor_*.log | grep "Backend.*heavydb"
```

#### Enable Dual-Write Mode
```bash
cd /mnt/optimizer_share

# Configure dual-write
sed -i 's/DATA_BACKEND=.*/DATA_BACKEND=heavydb/g' config/production_config.ini
sed -i 's/migration_phase=.*/migration_phase=dual_write/g' config/production_config.ini
sed -i 's/enable_dual_write=.*/enable_dual_write=true/g' config/production_config.ini
sed -i 's/primary_backend=.*/primary_backend=heavydb/g' config/production_config.ini
sed -i 's/secondary_backend=.*/secondary_backend=parquet/g' config/production_config.ini

# Restart and monitor
sudo systemctl restart samba_job_queue_processor
tail -f logs/job_processor_*.log | grep -E "Dual.*write|Backend.*validation"
```

#### Progressive Cut-Over to Parquet
```bash
cd /mnt/optimizer_share

# Set traffic percentage (e.g., 25% to Parquet)
sed -i 's/traffic_percentage_parquet=.*/traffic_percentage_parquet=25/g' config/production_config.ini
sed -i 's/migration_phase=.*/migration_phase=cut_over/g' config/production_config.ini

# Monitor traffic distribution
sudo systemctl restart samba_job_queue_processor
tail -f logs/job_processor_*.log | grep -E "Routing.*parquet|Traffic.*distribution"
```

#### Full Parquet Mode
```bash
cd /mnt/optimizer_share

# Switch to Parquet primary
sed -i 's/DATA_BACKEND=.*/DATA_BACKEND=parquet/g' config/production_config.ini
sed -i 's/migration_phase=.*/migration_phase=parquet_only/g' config/production_config.ini
sed -i 's/traffic_percentage_parquet=.*/traffic_percentage_parquet=100/g' config/production_config.ini

# Restart and verify
sudo systemctl restart samba_job_queue_processor
tail -f logs/job_processor_*.log | grep "Backend.*parquet"
```

## Missed Job Replay Procedures

### Detect Missed Jobs

#### Check for Failed Jobs
```bash
cd /mnt/optimizer_share

# Find failed jobs during migration
find jobs/failed/ -name "*.json" -newer migration_start.timestamp

# Check for incomplete processing
find jobs/processing/ -mmin +60 -name "*.json"

# Verify output completeness
find output/ -name "run_*" -type d | while read dir; do
    if [ ! -f "$dir/optimization_summary_*.txt" ]; then
        echo "Incomplete run: $dir"
    fi
done
```

#### Identify Backend Processing Gaps
```bash
# Check HeavyDB processing log gaps
grep -E "Processing.*heavydb" logs/job_processor_*.log | \
awk '{print $1, $2}' | sort | uniq -c | awk '$1 < 2 {print $2, $3}'

# Check Parquet processing log gaps  
grep -E "Processing.*parquet" logs/job_processor_*.log | \
awk '{print $1, $2}' | sort | uniq -c | awk '$1 < 2 {print $2, $3}'
```

### Replay Mechanisms

#### Manual Job Replay
```bash
cd /mnt/optimizer_share

# Create replay script
cat > scripts/replay_jobs.sh << 'EOF'
#!/bin/bash
REPLAY_DIR="/mnt/optimizer_share/jobs/replay"
QUEUE_DIR="/mnt/optimizer_share/jobs/queue"

# Process failed jobs
for job_file in jobs/failed/*.json; do
    if [ -f "$job_file" ]; then
        echo "Replaying: $(basename $job_file)"
        cp "$job_file" "$QUEUE_DIR/"
        rm "$job_file"
        sleep 2
    fi
done

# Process stuck jobs
for job_file in jobs/processing/*.json; do
    if [ -f "$job_file" ] && [ $(($(date +%s) - $(stat -c %Y "$job_file"))) -gt 3600 ]; then
        echo "Replaying stuck job: $(basename $job_file)"
        cp "$job_file" "$QUEUE_DIR/"
        rm "$job_file"
        sleep 2
    fi
done
EOF

chmod +x scripts/replay_jobs.sh
./scripts/replay_jobs.sh
```

#### Automated Gap Detection and Replay
```bash
cd /mnt/optimizer_share

# Create comprehensive replay system
cat > scripts/detect_and_replay_gaps.py << 'EOF'
#!/usr/bin/env python3
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

def detect_processing_gaps():
    """Detect jobs that may have been missed during backend switching"""
    base_path = Path("/mnt/optimizer_share")
    gaps = []
    
    # Check completed jobs vs expected outputs
    completed_path = base_path / "jobs" / "completed"
    output_path = base_path / "output"
    
    for job_file in completed_path.glob("*.json"):
        with open(job_file) as f:
            job_data = json.load(f)
        
        # Check if corresponding output exists
        timestamp = job_data.get('timestamp', '')
        expected_output = output_path / f"run_{timestamp}"
        
        if not expected_output.exists():
            gaps.append({
                'job_file': job_file,
                'expected_output': expected_output,
                'timestamp': timestamp
            })
    
    return gaps

def replay_missed_jobs(gaps):
    """Replay jobs that were missed"""
    queue_path = Path("/mnt/optimizer_share/jobs/queue")
    
    for gap in gaps:
        print(f"Replaying job: {gap['job_file'].name}")
        shutil.copy(gap['job_file'], queue_path / gap['job_file'].name)

if __name__ == "__main__":
    gaps = detect_processing_gaps()
    if gaps:
        print(f"Found {len(gaps)} processing gaps")
        replay_missed_jobs(gaps)
    else:
        print("No processing gaps detected")
EOF

python3 scripts/detect_and_replay_gaps.py
```

#### Batch Replay with Backend Selection
```bash
# Replay specific jobs with backend override
cat > scripts/replay_with_backend.sh << 'EOF'
#!/bin/bash
BACKEND=${1:-heavydb}  # Default to HeavyDB
REPLAY_COUNT=${2:-10}  # Default replay 10 jobs

cd /mnt/optimizer_share

# Temporarily override backend for replay
echo "Replaying jobs with backend: $BACKEND"
sed -i "s/DATA_BACKEND=.*/DATA_BACKEND=$BACKEND/g" config/production_config.ini

# Process failed jobs
count=0
for job_file in jobs/failed/*.json; do
    if [ -f "$job_file" ] && [ $count -lt $REPLAY_COUNT ]; then
        echo "Replaying: $(basename $job_file)"
        cp "$job_file" jobs/queue/
        rm "$job_file"
        count=$((count + 1))
        sleep 5  # Allow processing time
    fi
done

echo "Replayed $count jobs with $BACKEND backend"
EOF

chmod +x scripts/replay_with_backend.sh

# Usage examples:
./scripts/replay_with_backend.sh heavydb 20   # Replay 20 jobs with HeavyDB
./scripts/replay_with_backend.sh parquet 10   # Replay 10 jobs with Parquet
```

## Migration Phase Rollback Checkpoints

### Phase 1: Dual-Write Implementation Rollback

#### Checkpoint 1.1: Infrastructure Setup
```bash
# Rollback Condition: Dual-write infrastructure fails
cd /mnt/optimizer_share

echo "Rolling back Dual-Write Infrastructure..."

# Disable dual-write
sed -i 's/enable_dual_write=.*/enable_dual_write=false/g' config/production_config.ini
sed -i 's/migration_phase=.*/migration_phase=heavydb_only/g' config/production_config.ini

# Remove Parquet modules if causing issues
mv backend/lib/parquet_connector/ backend/lib/parquet_connector.disabled/
mv backend/lib/arrow_processor/ backend/lib/arrow_processor.disabled/

# Restart with HeavyDB only
sudo systemctl restart samba_job_queue_processor

# Validation
tail -f logs/job_processor_*.log | grep "heavydb.*only"
```

#### Checkpoint 1.2: Data Validation Framework Rollback
```bash
# Rollback Condition: Validation framework causes performance issues
cd /mnt/optimizer_share

echo "Rolling back Data Validation Framework..."

# Disable validation
sed -i 's/validation_enabled=.*/validation_enabled=false/g' config/production_config.ini
sed -i 's/rollback_on_validation_failure=.*/rollback_on_validation_failure=false/g' config/production_config.ini

# Comment out validation in workflow files
sed -i 's/^.*validate_results/#&/g' backend/csv_only_heavydb_workflow.py

sudo systemctl restart samba_job_queue_processor
```

#### Checkpoint 1.3: Performance Monitoring Rollback
```bash
# Rollback Condition: Monitoring overhead impacts performance
cd /mnt/optimizer_share

echo "Rolling back Performance Monitoring..."

# Disable migration monitoring
sed -i 's/track_migration_metrics=.*/track_migration_metrics=false/g' config/production_config.ini
sed -i 's/log_backend_performance=.*/log_backend_performance=false/g' config/production_config.ini

# Remove monitoring calls
sed -i 's/^.*monitor_backend_performance/#&/g' backend/lib/performance_monitoring/

sudo systemctl restart samba_job_queue_processor
```

### Phase 2: Cut-Over Implementation Rollback

#### Checkpoint 2.1: Traffic Routing Rollback
```bash
# Rollback Condition: Parquet backend shows errors or performance issues
cd /mnt/optimizer_share

echo "Rolling back Traffic Routing..."

# Force all traffic to HeavyDB
sed -i 's/traffic_percentage_parquet=.*/traffic_percentage_parquet=0/g' config/production_config.ini
sed -i 's/migration_phase=.*/migration_phase=dual_write/g' config/production_config.ini

# Verify routing
sudo systemctl restart samba_job_queue_processor
tail -f logs/job_processor_*.log | grep "Traffic.*100.*heavydb"
```

#### Checkpoint 2.2: Blue-Green Deployment Rollback
```bash
# Rollback Condition: Green environment (Parquet) fails
cd /mnt/optimizer_share

echo "Rolling back to Blue Environment (HeavyDB)..."

# Switch symlinks back to blue
ln -sf csv_only_heavydb_workflow.py backend/current_workflow.py

# Update configuration
sed -i 's/DATA_BACKEND=.*/DATA_BACKEND=heavydb/g' config/production_config.ini
sed -i 's/deployment_environment=.*/deployment_environment=blue/g' config/production_config.ini

sudo systemctl restart samba_job_queue_processor
```

#### Checkpoint 2.3: Data Consistency Rollback
```bash
# Rollback Condition: Data consistency issues detected
cd /mnt/optimizer_share

echo "Rolling back due to Data Consistency Issues..."

# Stop all processing
sudo systemctl stop samba_job_queue_processor

# Restore from last known good state
if [ -f data/backup/last_good_state.tar.gz ]; then
    tar -xzf data/backup/last_good_state.tar.gz -C /
fi

# Force validation mode
sed -i 's/DATA_BACKEND=.*/DATA_BACKEND=heavydb/g' config/production_config.ini
sed -i 's/enable_strict_validation=.*/enable_strict_validation=true/g' config/production_config.ini

sudo systemctl start samba_job_queue_processor
```

### Phase 3: Full Migration Rollback

#### Checkpoint 3.1: Performance Regression Rollback
```bash
# Rollback Condition: Parquet system shows worse performance than HeavyDB
cd /mnt/optimizer_share

echo "Rolling back due to Performance Regression..."

# Full rollback to HeavyDB
sed -i 's/DATA_BACKEND=.*/DATA_BACKEND=heavydb/g' config/production_config.ini
sed -i 's/migration_phase=.*/migration_phase=heavydb_only/g' config/production_config.ini
sed -i 's/enable_dual_write=.*/enable_dual_write=false/g' config/production_config.ini

# Disable Parquet components
mv backend/lib/parquet_connector/ backend/lib/parquet_connector.disabled/
mv backend/lib/cudf_accelerator/ backend/lib/cudf_accelerator.disabled/

sudo systemctl restart samba_job_queue_processor
```

#### Checkpoint 3.2: System Stability Rollback
```bash
# Rollback Condition: System stability issues with Parquet
cd /mnt/optimizer_share

echo "Rolling back due to System Stability Issues..."

# Emergency shutdown
sudo systemctl stop samba_job_queue_processor

# Restore configuration backup
cp config/production_config.ini.heavydb_backup config/production_config.ini

# Restore system state
if [ -f /tmp/system_state_backup.tar.gz ]; then
    tar -xzf /tmp/system_state_backup.tar.gz -C /mnt/optimizer_share/
fi

sudo systemctl start samba_job_queue_processor
```

## Emergency Procedures for Production Issues

### Critical System Failure

#### Level 1: Service Restart
```bash
# Quick restart for minor issues
cd /mnt/optimizer_share

sudo systemctl restart samba_job_queue_processor
sleep 10

# Check if service is healthy
if ! systemctl is-active --quiet samba_job_queue_processor; then
    echo "Service restart failed, escalating to Level 2"
fi
```

#### Level 2: Configuration Reset
```bash
# Reset to known good configuration
cd /mnt/optimizer_share

# Backup current config
cp config/production_config.ini config/production_config.ini.failed_backup

# Restore known good config
cp config/production_config.ini.last_known_good config/production_config.ini

# Reset backend to HeavyDB
sed -i 's/DATA_BACKEND=.*/DATA_BACKEND=heavydb/g' config/production_config.ini

sudo systemctl restart samba_job_queue_processor
```

#### Level 3: Emergency Rollback
```bash
# Full emergency rollback
cd /mnt/optimizer_share

./scripts/emergency_rollback.sh

# If script fails, manual emergency procedure:
sudo systemctl stop samba_job_queue_processor

# Force HeavyDB configuration
cat > config/production_config.ini.emergency << 'EOF'
[SYSTEM]
DATA_BACKEND = heavydb
migration_phase = heavydb_only
gpu_enabled = true

[MIGRATION]
enable_dual_write = false
validation_enabled = false
EOF

cp config/production_config.ini.emergency config/production_config.ini
sudo systemctl start samba_job_queue_processor
```

### Data Corruption Recovery

#### Detect Data Corruption
```bash
cd /mnt/optimizer_share

# Check for corrupted output files
find output/ -name "*.txt" -size 0 -o -name "*.csv" -size 0

# Validate job completion integrity
python3 scripts/validate_job_integrity.py

# Check database consistency (if applicable)
if [ "$DATA_BACKEND" = "heavydb" ]; then
    ./scripts/check_heavydb_integrity.sh
fi
```

#### Recovery Procedures
```bash
cd /mnt/optimizer_share

# Stop processing
sudo systemctl stop samba_job_queue_processor

# Restore from backup if available
if [ -f data/backup/daily_backup.tar.gz ]; then
    echo "Restoring from daily backup..."
    tar -xzf data/backup/daily_backup.tar.gz -C /
fi

# Replay corrupted jobs
find jobs/completed/ -name "*.json" -newer data/corruption_detected.timestamp | while read job; do
    cp "$job" jobs/queue/
done

sudo systemctl start samba_job_queue_processor
```

### Network Storage Issues

#### Samba Share Recovery
```bash
# Check Samba share accessibility
cd /mnt/optimizer_share

# Test network connectivity
ping -c 3 204.12.223.93

# Test Samba mount
if ! mountpoint -q /mnt/optimizer_share; then
    echo "Samba share not mounted, attempting remount..."
    sudo mount -t cifs //204.12.223.93/optimizer_share /mnt/optimizer_share \
        -o username=opt_admin,password=Chetti@123,uid=1000,gid=1000
fi

# Verify write access
touch /mnt/optimizer_share/test_write && rm /mnt/optimizer_share/test_write
```

#### Windows Client Connectivity Issues
```bash
# Check for client accessibility issues
cd /mnt/optimizer_share

# Verify Samba service status
sudo systemctl status smbd nmbd

# Check client connections
sudo smbstatus

# Monitor access logs
tail -f /var/log/samba/log.smbd | grep -E "connect|disconnect"

# Test from Linux client perspective
smbclient -L //204.12.223.93 -U opt_admin
```

## Validation and Monitoring

### Backend Health Checks
```bash
# Automated health check script
cat > scripts/backend_health_check.sh << 'EOF'
#!/bin/bash
cd /mnt/optimizer_share

echo "=== Backend Health Check ==="
echo "Date: $(date)"

# Check current backend
BACKEND=$(grep "DATA_BACKEND" config/production_config.ini | cut -d'=' -f2 | tr -d ' ')
echo "Current Backend: $BACKEND"

# Check service status
echo "Service Status: $(systemctl is-active samba_job_queue_processor)"

# Check job queue
QUEUE_COUNT=$(ls jobs/queue/*.json 2>/dev/null | wc -l)
PROCESSING_COUNT=$(ls jobs/processing/*.json 2>/dev/null | wc -l)
echo "Jobs in Queue: $QUEUE_COUNT"
echo "Jobs Processing: $PROCESSING_COUNT"

# Check recent completions
RECENT_COMPLETED=$(find jobs/completed/ -mmin -10 -name "*.json" | wc -l)
echo "Recent Completions (10min): $RECENT_COMPLETED"

# Check for errors
ERROR_COUNT=$(grep -c "ERROR" logs/job_processor_*.log 2>/dev/null || echo 0)
echo "Recent Errors: $ERROR_COUNT"

# Performance check
if [ -f output/run_*/optimization_summary_*.txt ]; then
    LAST_RUNTIME=$(grep "Execution Time" output/run_*/optimization_summary_*.txt | tail -1 | awk '{print $3}')
    echo "Last Runtime: ${LAST_RUNTIME}s"
fi

echo "=== End Health Check ==="
EOF

chmod +x scripts/backend_health_check.sh
./scripts/backend_health_check.sh
```

### Migration Progress Monitoring
```bash
# Migration progress dashboard
cat > scripts/migration_dashboard.py << 'EOF'
#!/usr/bin/env python3
import json
import time
from pathlib import Path
from collections import defaultdict

def monitor_migration_progress():
    base_path = Path("/mnt/optimizer_share")
    
    while True:
        print("\n" + "="*60)
        print("MIGRATION PROGRESS DASHBOARD")
        print("="*60)
        
        # Backend status
        with open(base_path / "config" / "production_config.ini") as f:
            config = f.read()
            backend = next(line.split('=')[1].strip() for line in config.split('\n') if 'DATA_BACKEND' in line)
            phase = next(line.split('=')[1].strip() for line in config.split('\n') if 'migration_phase' in line)
        
        print(f"Current Backend: {backend}")
        print(f"Migration Phase: {phase}")
        
        # Job processing stats
        job_stats = defaultdict(int)
        for status in ['queue', 'processing', 'completed', 'failed']:
            job_files = list((base_path / "jobs" / status).glob("*.json"))
            job_stats[status] = len(job_files)
        
        print(f"Jobs - Queue: {job_stats['queue']}, Processing: {job_stats['processing']}")
        print(f"Jobs - Completed: {job_stats['completed']}, Failed: {job_stats['failed']}")
        
        # Recent performance
        recent_outputs = list((base_path / "output").glob("run_*"))
        if recent_outputs:
            latest_run = max(recent_outputs, key=lambda p: p.stat().st_mtime)
            print(f"Latest Run: {latest_run.name}")
        
        time.sleep(30)

if __name__ == "__main__":
    monitor_migration_progress()
EOF

python3 scripts/migration_dashboard.py
```

## Contact Information and Escalation

### Primary Contacts
- **System Administrator**: `ops-team@company.com`
- **Backend Team Lead**: `backend-lead@company.com`  
- **DevOps Engineer**: `devops@company.com`
- **Emergency Hotline**: `+1-555-EMERGENCY`

### Escalation Matrix

| Severity | Response Time | Escalation Path |
|----------|---------------|-----------------|
| P1 - System Down | 15 minutes | Ops Team → Backend Lead → DevOps Manager |
| P2 - Performance Degradation | 1 hour | Backend Team → System Admin |
| P3 - Data Validation Issues | 4 hours | Backend Team |
| P4 - Migration Phase Issues | 24 hours | Backend Team |

### Emergency Communication Channels
- **Slack**: `#heavy-optimizer-emergency`
- **PagerDuty**: Heavy Optimizer Platform service
- **Phone Tree**: Available in company directory

## Documentation References

- **Architecture**: `/mnt/optimizer_share/docs/architecture.md`
- **Migration Plan**: `/mnt/optimizer_share/docs/migration_plan.md`
- **Configuration Guide**: `/mnt/optimizer_share/docs/CONFIG_MANAGEMENT_GUIDE.md`
- **System Logs**: `/mnt/optimizer_share/logs/`

---

**Document Control**
- **Last Updated**: August 2, 2025
- **Review Cycle**: Monthly during migration, quarterly post-migration
- **Approval**: PO Agent, Backend Team Lead, DevOps Manager