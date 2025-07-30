# Story: Multi-Format Output Generation Engine

**As a** User,
**I want to** receive optimization results in multiple formats as shown in the architecture,
**So that** I can analyze results using different tools and perspectives.

**Acceptance Criteria**:

Generate all 8 output types from the architecture diagram:
1.  **ULTA Inversion Report (.md)**: Details on inverted strategies and their performance improvements.
2.  **Zone Analysis Report (.md)**: A breakdown of performance for each trading zone.
3.  **Performance Report (.txt)**: A text-based summary of the final portfolio's key metrics.
4.  **Portfolio Composition (.csv)**: A list of the selected strategies in the final portfolio.
5.  **Excel Summary (.xlsx)**: A multi-sheet spreadsheet with comprehensive, zone-based analysis.
6.  **Execution Summary (.json)**: A JSON file containing all metadata and results from the run.
7.  **Equity Curves (.png)**: A chart showing the equity curve of the final portfolio's performance.
8.  **Algorithm Comparison (.png)**: A chart comparing the performance of all 8 optimization algorithms.

Each output must:
- Use a template for consistent formatting.
- Include a timestamp and other job metadata.
- Be saved to the correct output directory.
- Handle any data generation errors gracefully.
