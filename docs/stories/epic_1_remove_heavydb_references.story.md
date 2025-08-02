# Epic 1 Story: Remove HeavyDB References

## Status
To Do

## Story
**As a** Product Manager and System Architect,
**I want** to audit all stories and documentation to deprecate or rewrite HeavyDB-specific tasks,
**so that** the platform documentation accurately reflects the Parquet/Arrow/cuDF migration and eliminates legacy technology references

## Acceptance Criteria
1. All story files updated or cancelled to remove HeavyDB-specific tasks
2. `grep -Ri 'HeavyDB' docs/stories` returns only migration-context lines (references explaining what was replaced)
3. Documentation cleanup tracking shows 100% completion of HeavyDB reference removal

## Tasks / Subtasks

### Task 1: Audit Current HeavyDB References (AC: 2)
- [ ] Subtask 1.1: Complete inventory of HeavyDB references across all documentation
  - [ ] Scan all 13 story files with 132 total HeavyDB occurrences
  - [ ] Categorize references: active tasks vs historical context vs migration notes
  - [ ] Create priority matrix for reference handling (deprecate, rewrite, preserve as context)
  - [ ] Document findings in tracking spreadsheet
- [ ] Subtask 1.2: Analyze reference types and replacement strategy
  - [ ] Identify HeavyDB-specific technical tasks that need Parquet/Arrow/cuDF equivalents
  - [ ] Mark historical references that should be preserved for migration context
  - [ ] Flag deprecated stories that should be cancelled or archived
  - [ ] Create replacement task templates for common HeavyDB operations

### Task 2: Update Active Stories with HeavyDB Dependencies (AC: 1)
- [ ] Subtask 2.1: Rewrite Story 1.1 HeavyDB workflow references
  - [ ] Update `/docs/stories/1.1.story.md` - Replace 6 HeavyDB references in workflow integration
  - [ ] Change `csv_only_heavydb_workflow.py` references to `parquet_cudf_workflow.py`
  - [ ] Update data loading specifications from HeavyDB to Parquet/Arrow/cuDF
  - [ ] Preserve algorithm implementation references (no change needed)
- [ ] Subtask 2.2: Rewrite Story 1.3 GPU acceleration references  
  - [ ] Update `/docs/stories/1.3.story.md` - Replace 62 HeavyDB references in GPU optimization
  - [ ] Change HeavyDB GPU acceleration to cuDF GPU operations
  - [ ] Update performance metrics from HeavyDB baseline to cuDF baseline
  - [ ] Rewrite correlation calculation tasks for cuDF implementation
- [ ] Subtask 2.3: Rewrite Story 1.4 workflow integration references
  - [ ] Update `/docs/stories/1.4.story.md` - Replace 16 HeavyDB references in complete workflow
  - [ ] Change end-to-end workflow from HeavyDB to Parquet/Arrow/cuDF stack
  - [ ] Update output generation and report creation for new data pipeline
  - [ ] Modify integration testing approach for new technology stack
- [ ] Subtask 2.4: Update Story 2.1 enhanced optimization references
  - [ ] Update `/docs/stories/2.1.story.md` - Replace 12 HeavyDB references in financial optimization
  - [ ] Change advanced analytics from HeavyDB queries to cuDF operations
  - [ ] Update Kelly Criterion implementation for cuDF data structures
  - [ ] Modify risk metrics calculation for new pipeline

### Task 3: Archive or Cancel Deprecated Stories (AC: 1)
- [ ] Subtask 3.1: Handle Epic-1 documentation HeavyDB references
  - [ ] Review `/docs/stories/epic-1/` - 10 total HeavyDB references across 6 files
  - [ ] Update executive summary to reflect Parquet/Arrow/cuDF migration completion
  - [ ] Modify success metrics from HeavyDB performance to cuDF performance baselines
  - [ ] Update resource requirements for new technology stack
  - [ ] Archive original epic scope as historical context
- [ ] Subtask 3.2: Cancel or rewrite Epic 1 supporting stories
  - [ ] Review `epic_1_ci_cd_automated_tests.story.md` - 2 HeavyDB references
  - [ ] Review `epic_1_performance_benchmarking_framework.story.md` - 1 HeavyDB reference  
  - [ ] Update CI/CD testing frameworks to use Parquet/Arrow/cuDF instead of HeavyDB
  - [ ] Modify performance benchmarking to compare cuDF vs CPU operations
- [ ] Subtask 3.3: Handle archived story references in 31Jul25_stories
  - [ ] Review archived stories in `/docs/31Jul25_stories/` for HeavyDB cleanup needs
  - [ ] Mark archived stories as "Historical - Pre-Migration" to preserve context
  - [ ] Ensure no active tasks reference deprecated HeavyDB stories
  - [ ] Create migration notes explaining technology evolution

### Task 4: Update Configuration and Technical References (AC: 1, 2)
- [ ] Subtask 4.1: Update workflow and connector references
  - [ ] Replace `lib.heavydb_connector` references with `lib.parquet_pipeline` and `lib.cudf_engine`
  - [ ] Update `csv_only_heavydb_workflow.py` filename references to `parquet_cudf_workflow.py`
  - [ ] Modify data loading function references from HeavyDB to Parquet/Arrow
  - [ ] Update SQL query references to cuDF DataFrame operations
- [ ] Subtask 4.2: Revise performance and optimization references
  - [ ] Change HeavyDB GPU acceleration references to cuDF GPU operations
  - [ ] Update memory limitation references (32GB HeavyDB → unlimited cuDF scaling)
  - [ ] Modify correlation calculation references for cuDF implementation
  - [ ] Update fitness calculation references from HeavyDB aggregations to cuDF groupby
- [ ] Subtask 4.3: Create migration context preservation
  - [ ] Add "Migration Note" sections to updated stories explaining HeavyDB → cuDF transition
  - [ ] Preserve historical performance baselines for comparison purposes
  - [ ] Document technology stack evolution for future reference
  - [ ] Maintain audit trail of what was changed and why

### Task 5: Validation and Testing Framework Update (AC: 2, 3)
- [ ] Subtask 5.1: Execute comprehensive reference audit
  - [ ] Run `grep -Ri 'HeavyDB' docs/stories` to verify cleanup completion
  - [ ] Ensure only migration-context lines remain (target: <10 historical references)
  - [ ] Validate no active tasks reference deprecated HeavyDB functionality
  - [ ] Create before/after comparison report showing reference reduction
- [ ] Subtask 5.2: Validate story consistency and completeness
  - [ ] Ensure all updated stories maintain logical flow and technical accuracy
  - [ ] Verify replacement tasks provide equivalent functionality using new stack
  - [ ] Confirm no broken internal references between stories
  - [ ] Test that updated stories align with `epic_1_parquet_arrow_cudf.story.md`
- [ ] Subtask 5.3: Documentation quality assurance
  - [ ] Review all updated stories for technical accuracy and clarity
  - [ ] Ensure migration notes are consistent across all updated documentation
  - [ ] Verify acceptance criteria still achievable with new technology stack
  - [ ] Create cleanup completion report with metrics and change log

## Dev Notes

### Previous Story Context
This story supports the Epic 1 Parquet/Arrow/cuDF migration by cleaning up legacy HeavyDB references that would confuse developers and create technical debt. The migration is already defined in `epic_1_parquet_arrow_cudf.story.md` but requires comprehensive documentation cleanup to complete.

**Current HeavyDB Reference Distribution:**
- Story 1.1: 6 references (workflow integration)  
- Story 1.3: 62 references (GPU acceleration)
- Story 1.4: 16 references (complete workflow)
- Story 2.1: 12 references (enhanced optimization)
- Epic-1 docs: 10 references across 6 files
- Supporting stories: 3 references (CI/CD + benchmarking)
- Other stories: 23 references across remaining files

### Data Models
No data model changes required - this is a documentation cleanup story. However, documentation updates must reflect the new data pipeline:

**Previous HeavyDB References**:
```
HeavyDB connection → lib.heavydb_connector
SQL queries → HeavyDB aggregations  
csv_only_heavydb_workflow.py → Main workflow
32GB memory limits → HeavyDB constraints
```

**Replacement Parquet/Arrow/cuDF References**:
```
Parquet storage → lib.parquet_pipeline
cuDF operations → GPU DataFrame operations
parquet_cudf_workflow.py → Main workflow  
Unlimited scaling → cuDF GPU memory management
```

### API Specifications
No API changes - documentation updates only. Updated stories must reference new component interfaces:

```python
# Old HeavyDB references to remove
lib.heavydb_connector.load_strategy_data()
lib.heavydb_connector.calculate_correlations_gpu()
csv_only_heavydb_workflow.run_optimization()

# New Parquet/Arrow/cuDF references to add
lib.parquet_pipeline.load_parquet_to_cudf()
lib.cudf_engine.calculate_correlations_cudf()
parquet_cudf_workflow.run_optimization()
```

### Component Specifications
**Documentation Components to Update**:
- Story files: `/docs/stories/*.story.md`
- Epic documentation: `/docs/stories/epic-1/*.md`
- Archive handling: `/docs/31Jul25_stories/*heavydb*.md`
- Configuration docs: Any HeavyDB connector references

**Replacement Strategy**:
1. **Active Task References**: Rewrite for Parquet/Arrow/cuDF equivalents
2. **Historical Context**: Preserve with "Migration Note" annotations  
3. **Deprecated Features**: Mark as "Historical - Pre-Migration"
4. **Performance Baselines**: Update from HeavyDB metrics to cuDF metrics

### File Locations
**Primary Story Files to Update**:
- `/docs/stories/1.1.story.md` - Workflow integration (6 refs)
- `/docs/stories/1.3.story.md` - GPU acceleration (62 refs)  
- `/docs/stories/1.4.story.md` - Complete workflow (16 refs)
- `/docs/stories/2.1.story.md` - Enhanced optimization (12 refs)

**Supporting Documentation**:
- `/docs/stories/epic-1/*.md` - Epic documentation (10 refs)
- `/docs/stories/epic_1_*.story.md` - Supporting epic stories (3 refs)
- `/docs/31Jul25_stories/story_heavydb_*.md` - Archived stories (historical)

**New Documentation to Reference**:
- `/docs/stories/epic_1_parquet_arrow_cudf.story.md` - Migration implementation
- Updated architecture docs reflecting new technology stack
- Configuration files for Parquet/Arrow/cuDF settings

### Testing Requirements
**Documentation Validation**:
```bash
# Primary acceptance criteria validation
grep -Ri 'HeavyDB' docs/stories/
# Should return <10 migration-context references only

# Story consistency checks  
grep -r "csv_only_heavydb_workflow" docs/stories/
# Should return 0 results (all updated to parquet_cudf_workflow)

# Cross-reference validation
grep -r "lib.heavydb_connector" docs/stories/  
# Should return 0 results (all updated to new components)
```

**Quality Assurance Process**:
1. Before/after reference count comparison
2. Story logic flow validation after updates
3. Technical accuracy review for new technology references
4. Consistency check across all updated documentation

### Technical Constraints
**Documentation Standards**:
- Maintain existing story format and structure [Source: Story template in 1.1.story.md]
- Preserve historical context for auditing and migration tracking
- Ensure technical accuracy for new Parquet/Arrow/cuDF references
- Maintain consistency with Epic 1 migration implementation plan

**Migration Context Preservation**:
- Add "Migration Note" sections explaining HeavyDB → cuDF transition
- Preserve original performance baselines for comparison
- Maintain audit trail of documentation changes
- Ensure no loss of institutional knowledge about system evolution

**Change Management**:
- Update change logs for all modified stories
- Create comprehensive cleanup completion report
- Document rationale for each type of reference handling
- Ensure rollback capability if documentation changes create confusion

## Change Log
| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-08-02 | 1.0 | Initial brownfield story creation for HeavyDB reference cleanup | PM Agent |

## Dev Agent Record

### Agent Model Used
[To be filled by Dev Agent]

### Debug Log References
[To be updated during implementation]

### Completion Notes List
[To be updated during implementation]

### File List
[To be updated during implementation]

## QA Results

### QA Review Summary
[To be completed by QA Agent after implementation]

### Acceptance Criteria Verification
[To be completed by QA Agent]

### Test Results
[To be completed by QA Agent]

### Code Quality Assessment
[To be completed by QA Agent]

### Issues Found
[To be completed by QA Agent]

### Compliance Check
[To be completed by QA Agent]

### Final QA Verdict
[To be completed by QA Agent]