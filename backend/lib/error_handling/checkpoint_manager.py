"""
Checkpoint Manager for State Persistence and Recovery

Provides checkpoint/restore capability for the optimization pipeline,
allowing jobs to resume from the last successful step after failures.
"""

import os
import json
import pickle
import shutil
import hashlib
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging

class CheckpointManager:
    """Manages checkpoint creation, storage, and restoration for pipeline state"""
    
    def __init__(self, base_dir: str = "/mnt/optimizer_share/checkpoints", 
                 job_id: Optional[str] = None):
        """
        Initialize CheckpointManager
        
        Args:
            base_dir: Base directory for checkpoint storage
            job_id: Unique job identifier for checkpoint isolation
        """
        self.base_dir = Path(base_dir)
        self.job_id = job_id or datetime.now().strftime("job_%Y%m%d_%H%M%S")
        self.checkpoint_dir = self.base_dir / self.job_id
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self.logger = logging.getLogger(__name__)
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        self.metadata = self._load_metadata()
        
    def save_checkpoint(self, state: Dict[str, Any], checkpoint_name: str,
                       description: str = "") -> bool:
        """
        Save current state as a checkpoint
        
        Args:
            state: Dictionary containing the current state to save
            checkpoint_name: Name for this checkpoint
            description: Optional description of the checkpoint
            
        Returns:
            bool: True if checkpoint saved successfully
        """
        try:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.checkpoint"
            
            # Create checkpoint data
            checkpoint_data = {
                'state': state,
                'timestamp': datetime.now().isoformat(),
                'name': checkpoint_name,
                'description': description,
                'checksum': self._calculate_checksum(state)
            }
            
            # Save checkpoint with both pickle (for complex objects) and JSON (for metadata)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Save JSON version for inspection/debugging
            json_path = self.checkpoint_dir / f"{checkpoint_name}.json"
            try:
                with open(json_path, 'w') as f:
                    json.dump(self._serialize_for_json(checkpoint_data), f, indent=2)
            except:
                # JSON serialization is optional, for debugging only
                pass
            
            # Update metadata
            self.metadata['checkpoints'][checkpoint_name] = {
                'timestamp': checkpoint_data['timestamp'],
                'description': description,
                'file': str(checkpoint_path),
                'checksum': checkpoint_data['checksum']
            }
            self._save_metadata()
            
            self.logger.info(f"Checkpoint saved: {checkpoint_name} at {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_name}: {e}", 
                            exc_info=True)
            return False
    
    def load_checkpoint(self, checkpoint_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load state from a checkpoint
        
        Args:
            checkpoint_name: Name of checkpoint to load (None for latest)
            
        Returns:
            Restored state dictionary or None if failed
        """
        try:
            # Get latest checkpoint if none specified
            if checkpoint_name is None:
                checkpoint_name = self.get_latest_checkpoint()
                if checkpoint_name is None:
                    self.logger.warning("No checkpoints available to load")
                    return None
            
            checkpoint_info = self.metadata['checkpoints'].get(checkpoint_name)
            if not checkpoint_info:
                self.logger.error(f"Checkpoint not found: {checkpoint_name}")
                return None
            
            checkpoint_path = Path(checkpoint_info['file'])
            if not checkpoint_path.exists():
                self.logger.error(f"Checkpoint file missing: {checkpoint_path}")
                return None
            
            # Load checkpoint
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Verify checksum
            if self._calculate_checksum(checkpoint_data['state']) != checkpoint_data['checksum']:
                self.logger.error(f"Checkpoint checksum mismatch: {checkpoint_name}")
                return None
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_name} from {checkpoint_data['timestamp']}")
            return checkpoint_data['state']
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_name}: {e}", 
                            exc_info=True)
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        for name, info in self.metadata['checkpoints'].items():
            checkpoints.append({
                'name': name,
                'timestamp': info['timestamp'],
                'description': info.get('description', ''),
                'file': info['file']
            })
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get name of the most recent checkpoint
        
        Returns:
            Name of latest checkpoint or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[0]['name']
        return None
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Delete a specific checkpoint
        
        Args:
            checkpoint_name: Name of checkpoint to delete
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            checkpoint_info = self.metadata['checkpoints'].get(checkpoint_name)
            if not checkpoint_info:
                return False
            
            # Delete files
            checkpoint_path = Path(checkpoint_info['file'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            json_path = self.checkpoint_dir / f"{checkpoint_name}.json"
            if json_path.exists():
                json_path.unlink()
            
            # Update metadata
            del self.metadata['checkpoints'][checkpoint_name]
            self._save_metadata()
            
            self.logger.info(f"Checkpoint deleted: {checkpoint_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint {checkpoint_name}: {e}", 
                            exc_info=True)
            return False
    
    def cleanup_old_checkpoints(self, keep_recent: int = 10) -> int:
        """
        Clean up old checkpoints, keeping only the most recent ones
        
        Args:
            keep_recent: Number of recent checkpoints to keep
            
        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep_recent:
            return 0
        
        deleted_count = 0
        for checkpoint in checkpoints[keep_recent:]:
            if self.delete_checkpoint(checkpoint['name']):
                deleted_count += 1
        
        return deleted_count
    
    def create_recovery_point(self, stage: str, state: Dict[str, Any]) -> bool:
        """
        Create a recovery point for a specific pipeline stage
        
        Args:
            stage: Name of the pipeline stage
            state: Current state to save
            
        Returns:
            bool: True if recovery point created successfully
        """
        checkpoint_name = f"recovery_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.save_checkpoint(state, checkpoint_name, f"Recovery point for {stage}")
    
    def get_recovery_points(self) -> List[Dict[str, Any]]:
        """
        Get all recovery points
        
        Returns:
            List of recovery point information
        """
        recovery_points = []
        for checkpoint in self.list_checkpoints():
            if checkpoint['name'].startswith('recovery_'):
                recovery_points.append(checkpoint)
        return recovery_points
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file or create new"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Create new metadata
        return {
            'job_id': self.job_id,
            'created': datetime.now().isoformat(),
            'checkpoints': {}
        }
    
    def _save_metadata(self) -> None:
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
    
    def _calculate_checksum(self, state: Dict[str, Any]) -> str:
        """Calculate checksum for state verification"""
        try:
            # Convert state to bytes for checksum
            state_bytes = pickle.dumps(state)
            return hashlib.sha256(state_bytes).hexdigest()
        except:
            return ""
    
    def _serialize_for_json(self, data: Any) -> Any:
        """Convert complex objects to JSON-serializable format"""
        if isinstance(data, dict):
            return {k: self._serialize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_for_json(v) for v in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            return str(data)