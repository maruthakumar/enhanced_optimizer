# Story: Financial Metrics Calculation Engine

**As a** Quant,
**I want to** calculate all financial metrics shown in the architecture diagram,
**So that** I can evaluate portfolio performance comprehensively.

**Acceptance Criteria**:

Calculate these specific metrics:
- Total ROI
- Maximum Drawdown
- Win Rate
- Profit Factor
- ROI/Drawdown Ratio (as the primary fitness metric)

**Requirements**:
- The implementation must use the exact calculation formulas from the legacy code to ensure consistency.
- It must support calculations at both the individual strategy level and the overall portfolio level.
- It must be able to handle different time periods for calculations.
- It must be able to generate metrics for intermediate results during optimization, not just the final portfolio.
