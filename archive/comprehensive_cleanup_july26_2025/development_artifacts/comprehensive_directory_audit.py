#!/usr/bin/env python3
"""
Comprehensive Directory Audit - Heavy Optimizer Platform
Analyzes entire directory structure and classifies files for reorganization
"""

import os
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class ComprehensiveDirectoryAuditor:
    def __init__(self):
        self.audit_results = {
            'audit_timestamp': datetime.now().isoformat(),
            'directory_structure': {},
            'file_classification': {},
            'false_claims_detection': {},
            'production_readiness': {},
            'archive_recommendations': {}
        }
        
        # False claims patterns to detect
        self.false_claims_patterns = [
            '24x speedup', '24x faster', '24 times faster',
            '96.8% efficiency', '96.8% parallel efficiency',
            'sub-second execution', 'sub second execution',
            'ultra-fast', 'ultra fast', 'lightning fast',
            'parallel efficiency', 'massive speedup',
            'dramatic performance improvement'
        ]
        
        # Obsolete file patterns
        self.obsolete_patterns = [
            'parallel_algorithm_orchestrator',
            'complete_production_workflow',
            'old_', 'deprecated_', 'backup_'
        ]
        
        # Production file patterns
        self.production_patterns = [
            'honest_production_workflow',
            'Enhanced_HeavyDB_Optimizer_Launcher',
            'Enhanced_HFT_Optimization',
            'Enhanced_Portfolio_Optimization',
            'output_generation_engine'
        ]
    
    def analyze_directory_structure(self, base_path: str) -> Dict[str, Any]:
        """Analyze complete directory structure"""
        print(f"🔍 Analyzing directory structure: {base_path}")
        
        structure = {
            'path': base_path,
            'total_files': 0,
            'total_size_bytes': 0,
            'subdirectories': {},
            'files': []
        }
        
        try:
            if not os.path.exists(base_path):
                return structure
            
            for root, dirs, files in os.walk(base_path):
                rel_path = os.path.relpath(root, base_path)
                
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        stat = os.stat(file_path)
                        file_info = {
                            'name': file,
                            'path': file_path,
                            'relative_path': os.path.relpath(file_path, base_path),
                            'size_bytes': stat.st_size,
                            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            'extension': os.path.splitext(file)[1].lower(),
                            'is_executable': os.access(file_path, os.X_OK)
                        }
                        
                        structure['files'].append(file_info)
                        structure['total_files'] += 1
                        structure['total_size_bytes'] += stat.st_size
                        
                    except (OSError, PermissionError) as e:
                        print(f"⚠️ Cannot access {file_path}: {e}")
            
            return structure
            
        except Exception as e:
            print(f"❌ Error analyzing {base_path}: {e}")
            return structure
    
    def classify_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a file based on content and patterns"""
        file_path = file_info['path']
        file_name = file_info['name']
        extension = file_info['extension']
        
        classification = {
            'category': 'unknown',
            'subcategory': 'unclassified',
            'production_ready': False,
            'contains_false_claims': False,
            'obsolete': False,
            'archive_recommended': False,
            'reason': ''
        }
        
        # Check for production patterns
        for pattern in self.production_patterns:
            if pattern.lower() in file_name.lower():
                classification['category'] = 'production'
                classification['production_ready'] = True
                classification['reason'] = f'Matches production pattern: {pattern}'
                break
        
        # Check for obsolete patterns
        for pattern in self.obsolete_patterns:
            if pattern.lower() in file_name.lower():
                classification['category'] = 'obsolete'
                classification['obsolete'] = True
                classification['archive_recommended'] = True
                classification['reason'] = f'Matches obsolete pattern: {pattern}'
                break
        
        # Classify by file type
        if extension in ['.py']:
            if classification['category'] == 'unknown':
                classification['category'] = 'python_module'
            classification['subcategory'] = 'python_script'
        elif extension in ['.md', '.txt', '.rst']:
            if classification['category'] == 'unknown':
                classification['category'] = 'documentation'
            classification['subcategory'] = 'documentation'
        elif extension in ['.bat', '.cmd']:
            if classification['category'] == 'unknown':
                classification['category'] = 'batch_file'
            classification['subcategory'] = 'windows_automation'
        elif extension in ['.xlsx', '.xls', '.csv']:
            classification['category'] = 'data'
            classification['subcategory'] = 'dataset'
        elif extension in ['.png', '.jpg', '.jpeg']:
            classification['category'] = 'output'
            classification['subcategory'] = 'visualization'
        elif extension in ['.json', '.log']:
            classification['category'] = 'metadata'
            classification['subcategory'] = 'system_data'
        
        # Check for false claims in text files
        if extension in ['.py', '.md', '.txt', '.rst', '.bat', '.cmd']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    for pattern in self.false_claims_patterns:
                        if pattern.lower() in content:
                            classification['contains_false_claims'] = True
                            classification['archive_recommended'] = True
                            classification['reason'] += f' Contains false claim: {pattern}'
                            break
            except Exception as e:
                print(f"⚠️ Cannot read {file_path}: {e}")
        
        return classification
    
    def detect_false_claims(self, directories: List[str]) -> Dict[str, List[Dict]]:
        """Detect files containing false performance claims"""
        print("🚨 Detecting false performance claims...")
        
        false_claims_files = {}
        
        for directory in directories:
            if not os.path.exists(directory):
                continue
                
            false_claims_files[directory] = []
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Only check text-based files
                    if not any(file.lower().endswith(ext) for ext in ['.py', '.md', '.txt', '.rst', '.bat', '.cmd']):
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        found_claims = []
                        for pattern in self.false_claims_patterns:
                            if pattern.lower() in content.lower():
                                # Find line numbers
                                lines = content.split('\n')
                                line_numbers = []
                                for i, line in enumerate(lines, 1):
                                    if pattern.lower() in line.lower():
                                        line_numbers.append(i)
                                
                                found_claims.append({
                                    'pattern': pattern,
                                    'line_numbers': line_numbers
                                })
                        
                        if found_claims:
                            false_claims_files[directory].append({
                                'file_path': file_path,
                                'relative_path': os.path.relpath(file_path, directory),
                                'false_claims': found_claims,
                                'file_size': os.path.getsize(file_path),
                                'modified_time': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                            })
                            
                    except Exception as e:
                        print(f"⚠️ Cannot scan {file_path}: {e}")
        
        return false_claims_files
    
    def generate_archive_recommendations(self, file_classifications: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate recommendations for archiving files"""
        print("📁 Generating archive recommendations...")
        
        recommendations = {
            'obsolete_modules': [],
            'false_claims_docs': [],
            'original_batch_files': [],
            'development_artifacts': [],
            'old_outputs': [],
            'keep_production': [],
            'keep_current_docs': []
        }
        
        for file_path, classification in file_classifications.items():
            if classification['archive_recommended']:
                if classification['contains_false_claims']:
                    if classification['subcategory'] == 'documentation':
                        recommendations['false_claims_docs'].append(file_path)
                    elif classification['subcategory'] == 'windows_automation':
                        recommendations['original_batch_files'].append(file_path)
                    elif classification['subcategory'] == 'python_script':
                        recommendations['obsolete_modules'].append(file_path)
                elif classification['obsolete']:
                    if classification['subcategory'] == 'python_script':
                        recommendations['obsolete_modules'].append(file_path)
                    else:
                        recommendations['development_artifacts'].append(file_path)
            elif classification['production_ready']:
                if classification['subcategory'] == 'python_script':
                    recommendations['keep_production'].append(file_path)
                elif classification['subcategory'] == 'documentation':
                    recommendations['keep_current_docs'].append(file_path)
        
        return recommendations
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run complete directory audit"""
        print("🚀 Starting Comprehensive Directory Audit")
        print("=" * 80)
        
        # Directories to audit
        directories_to_audit = [
            '/mnt/optimizer_share',
            '/home/administrator/Optimizer'
        ]
        
        # Analyze directory structures
        print("📊 Analyzing directory structures...")
        for directory in directories_to_audit:
            structure = self.analyze_directory_structure(directory)
            self.audit_results['directory_structure'][directory] = structure
            print(f"✅ {directory}: {structure['total_files']} files, {structure['total_size_bytes']/1024/1024:.1f}MB")
        
        # Classify all files
        print("\n🏷️ Classifying files...")
        all_file_classifications = {}
        
        for directory, structure in self.audit_results['directory_structure'].items():
            for file_info in structure['files']:
                classification = self.classify_file(file_info)
                all_file_classifications[file_info['path']] = classification
        
        self.audit_results['file_classification'] = all_file_classifications
        
        # Detect false claims
        print("\n🚨 Detecting false performance claims...")
        false_claims = self.detect_false_claims(directories_to_audit)
        self.audit_results['false_claims_detection'] = false_claims
        
        # Generate archive recommendations
        print("\n📁 Generating archive recommendations...")
        archive_recommendations = self.generate_archive_recommendations(all_file_classifications)
        self.audit_results['archive_recommendations'] = archive_recommendations
        
        # Production readiness assessment
        print("\n✅ Assessing production readiness...")
        production_files = [f for f, c in all_file_classifications.items() if c['production_ready']]
        obsolete_files = [f for f, c in all_file_classifications.items() if c['obsolete']]
        false_claim_files = [f for f, c in all_file_classifications.items() if c['contains_false_claims']]
        
        self.audit_results['production_readiness'] = {
            'production_ready_files': len(production_files),
            'obsolete_files': len(obsolete_files),
            'false_claim_files': len(false_claim_files),
            'total_files_audited': len(all_file_classifications),
            'cleanup_required': len(obsolete_files) + len(false_claim_files)
        }
        
        # Save audit results
        results_file = f"directory_audit_july26_2025_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.audit_results, f, indent=2)
        
        # Print summary
        self.print_audit_summary()
        
        print("=" * 80)
        print("🎉 COMPREHENSIVE DIRECTORY AUDIT COMPLETED")
        print(f"📄 Detailed results saved to: {results_file}")
        
        return self.audit_results
    
    def print_audit_summary(self):
        """Print audit summary"""
        print("\n📋 AUDIT SUMMARY")
        print("=" * 60)
        
        # Directory summary
        total_files = sum(s['total_files'] for s in self.audit_results['directory_structure'].values())
        total_size_mb = sum(s['total_size_bytes'] for s in self.audit_results['directory_structure'].values()) / 1024 / 1024
        
        print(f"📊 Total Files Audited: {total_files:,}")
        print(f"💾 Total Size: {total_size_mb:.1f}MB")
        
        # Classification summary
        classifications = self.audit_results['file_classification']
        categories = {}
        for classification in classifications.values():
            category = classification['category']
            categories[category] = categories.get(category, 0) + 1
        
        print(f"\n🏷️ File Categories:")
        for category, count in sorted(categories.items()):
            print(f"   {category}: {count} files")
        
        # False claims summary
        false_claims = self.audit_results['false_claims_detection']
        total_false_claim_files = sum(len(files) for files in false_claims.values())
        
        print(f"\n🚨 False Claims Detection:")
        print(f"   Files with false claims: {total_false_claim_files}")
        
        # Archive recommendations
        archive_recs = self.audit_results['archive_recommendations']
        total_archive_recommended = sum(len(files) for files in archive_recs.values())
        
        print(f"\n📁 Archive Recommendations:")
        print(f"   Files to archive: {total_archive_recommended}")
        print(f"   Production files to keep: {len(archive_recs['keep_production'])}")
        
        print("=" * 60)


def main():
    """Main execution function"""
    auditor = ComprehensiveDirectoryAuditor()
    results = auditor.run_comprehensive_audit()
    
    print(f"\n🎯 AUDIT COMPLETE")
    print(f"Files audited: {results['production_readiness']['total_files_audited']:,}")
    print(f"Cleanup required: {results['production_readiness']['cleanup_required']} files")
    print(f"Production ready: {results['production_readiness']['production_ready_files']} files")

if __name__ == "__main__":
    main()
