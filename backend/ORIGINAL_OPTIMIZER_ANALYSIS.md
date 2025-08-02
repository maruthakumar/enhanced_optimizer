# Original Zone Optimizer Analysis

## Algorithm Iteration Counts (from Optimizer_New_patched.py)

### Default Iteration Counts:
1. **Genetic Algorithm (GA)**: 50 generations
2. **Particle Swarm Optimization (PSO)**: 50 iterations  
3. **Simulated Annealing (SA)**: 1000 iterations
4. **Differential Evolution (DE)**: 50 iterations
5. **Ant Colony Optimization (ACO)**: 50 iterations
6. **Hill Climbing (HC)**: 200 iterations
7. **Bayesian Optimization (BO)**: 50 iterations

### Configuration Parameter:
- `ga_generations` from config (default: 50, minimum: 10)
- Used for GA, PSO, DE, ACO, and BO
- SA always uses 1000 iterations
- HC always uses 200 iterations

## Zone Naming Convention & Dynamic Data Creation

### Zone Weights:
```python
zone_weights = None
if "ZONE_WEIGHTS" in config:
    zone_weights_dict = {zone.lower().replace(" ", ""): float(weight) 
                         for zone, weight in config["ZONE_WEIGHTS"].items()}
    all_zones = sorted(set(zone.lower().replace(" ", "") for zone in unique_zones))
    zone_weights = np.array([zone_weights_dict.get(zone, 1.0) for zone in all_zones])
    zone_weights = zone_weights / np.sum(zone_weights)  # Normalize
```

### Zone Processing:
1. Zones are converted to lowercase and spaces removed
2. Weights are normalized to sum to 1.0
3. Default weight is 1.0 if not specified
4. Zone matrix shape: (days, zones, strategies)

### Zone Fitness Calculation:
```python
def evaluate_fitness_zone(individual, zone_matrix, metric, corr_matrix, zone_weights, drawdown_threshold):
    avg_returns = zone_matrix[:, :, individual].mean(axis=2)  # Average over selected strategies
    weighted_returns = np.dot(avg_returns, zone_weights)  # Apply zone weights
```

## Inversion Logic (ULTA)

### ULTA Strategy Detection:
1. Strategies with negative ROI are identified
2. Their returns are inverted (multiplied by -1)
3. New columns created with "_inv" suffix

### Inversion Report Generation:
```python
def generate_inversion_report(inverted_strategies):
    # Creates markdown report with:
    # - Original strategy name
    # - Original PnL (negative)
    # - Inverted PnL (positive)
    # - Original Ratio
    # - Inverted Ratio
```

### Inversion Fitness Handling:
```python
if roi < 0:
    # Negative ROI with minimal drawdown is bad
    fitness = roi * 10   # Still negative but not as extreme
```

## Correlation Implementation

### Correlation Matrix:
1. Calculated using numpy/pandas correlation
2. Used in fitness evaluation to penalize correlated strategies
3. Diagonal set to 0 (self-correlation ignored)

### Correlation Penalty:
```python
avg_corr = np.mean(corr_values)
normalized_corr = (avg_corr - MEAN_CORRELATION) / (MAX_CORRELATION - MEAN_CORRELATION)
normalized_corr = np.clip(normalized_corr, 0, 1)
```

## Key Differences from Our Implementation

### 1. Iteration Counts:
Our current implementation:
- GA: 100 generations (vs 50)
- PSO: 75 iterations (vs 50)
- SA: 1000 iterations (same)
- DE: 100 generations (vs 50)
- ACO: 50 iterations (same)
- HC: 100 iterations (vs 200)
- BO: Not configured (vs 50)

### 2. Zone Implementation:
- Original uses zone weights applied to returns
- Zone names are normalized (lowercase, no spaces)
- Zone-specific optimization with weighted fitness

### 3. Inversion Logic:
- Original generates detailed markdown reports
- Tracks original vs inverted metrics
- Creates new columns with "_inv" suffix

### 4. Missing in Our Implementation:
- Zone-based mode with weighted returns
- Inversion report generation
- Zone equity curve plotting
- Checkpoint/resume functionality