"""
Version history management for Paper2Slides.
Saves working versions after successful PDF compilations.
Allows reverting to previous working versions when LLM edits fail.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import logging
import shutil


class VersionHistory:
    """Manages version history for a paper's slides - saves only after successful compiles."""
    
    def __init__(self, paper_id: str):
        """
        Initialize version history for a specific paper.
        
        Args:
            paper_id: The paper ID (e.g., arxiv ID or generated PDF ID)
        """
        self.paper_id = paper_id
        self.history_dir = Path(f"source/{paper_id}/edit_history")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
    def save_version(self, tex_content: str, description: str = "Successful compile") -> bool:
        """
        Save a new version after successful compilation.
        
        Args:
            tex_content: The LaTeX content to save
            description: Description of this version
            
        Returns:
            True if successful, False otherwise
        """
        try:
            timestamp = datetime.now()
            # Use timestamp as filename for easy sorting
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            
            version_data = {
                'timestamp': timestamp.isoformat(),
                'description': description,
                'tex_content': tex_content
            }
            
            version_file = self.history_dir / f"version_{timestamp_str}.json"
            with open(version_file, 'w', encoding='utf-8') as f:
                json.dump(version_data, f, indent=2)
            
            logging.info(f"Saved version: {description} at {timestamp_str}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save version: {e}")
            return False
    
    def list_versions(self) -> List[Dict]:
        """
        List all saved versions, newest first.
        
        Returns:
            List of version dictionaries with metadata (without full tex_content)
        """
        versions = []
        if not self.history_dir.exists():
            return versions
        
        for version_file in sorted(self.history_dir.glob("version_*.json"), reverse=True):
            try:
                with open(version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    versions.append({
                        'filename': version_file.name,
                        'timestamp': data.get('timestamp', ''),
                        'description': data.get('description', 'Unknown'),
                    })
            except Exception as e:
                logging.error(f"Failed to read version file {version_file}: {e}")
        
        return versions
    
    def get_latest_version(self) -> Optional[str]:
        """
        Get the most recent working version.
        
        Returns:
            LaTeX content of the latest version, or None if no history exists
        """
        versions = self.list_versions()
        if not versions:
            return None
        
        latest_file = self.history_dir / versions[0]['filename']
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('tex_content')
        except Exception as e:
            logging.error(f"Failed to read latest version: {e}")
            return None
    
    def get_version_by_filename(self, filename: str) -> Optional[str]:
        """
        Get a specific version by filename.
        
        Args:
            filename: The version filename (e.g., "version_20250107_143022.json")
            
        Returns:
            LaTeX content if found, None otherwise
        """
        version_file = self.history_dir / filename
        if not version_file.exists():
            return None
        
        try:
            with open(version_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('tex_content')
        except Exception as e:
            logging.error(f"Failed to read version {filename}: {e}")
            return None
    
    def restore_version(self, filename: str, slides_tex_path: str) -> bool:
        """
        Restore a specific version to slides.tex.
        
        Args:
            filename: The version filename to restore
            slides_tex_path: Path to slides.tex file
            
        Returns:
            True if successful, False otherwise
        """
        content = self.get_version_by_filename(filename)
        if content is None:
            logging.error(f"Version {filename} not found")
            return False
        
        try:
            with open(slides_tex_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"Restored version {filename} to {slides_tex_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to restore version: {e}")
            return False
    
    def has_history(self) -> bool:
        """Check if any version history exists."""
        return len(self.list_versions()) > 0
    
    def clear_history(self) -> bool:
        """
        Clear all version history for this paper.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.history_dir.exists():
                shutil.rmtree(self.history_dir)
            self.history_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Cleared history for paper {self.paper_id}")
            return True
        except Exception as e:
            logging.error(f"Failed to clear history: {e}")
            return False


def get_history_manager(paper_id: str) -> VersionHistory:
    """
    Get a version history manager for a specific paper.
    
    Args:
        paper_id: The paper ID
        
    Returns:
        VersionHistory instance
    """
    return VersionHistory(paper_id)
