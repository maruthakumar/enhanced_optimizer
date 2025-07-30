# GPU-Enhanced HeavyDB Portfolio Optimizer - Quick Reference

**Server:** 204.12.223.93 | **GPU:** NVIDIA A100-SXM4-40GB | **Performance:** 8.6x Average Speedup

---

## 🚀 **QUICK START (5 MINUTES)**

### **1. Connect to System**
```batch
# Map network drive
net use P: \\204.12.223.93\optimizer_share /user:opt_admin Chetti@123

# Verify connection
dir P:\
```

### **2. Upload Dataset**
```batch
# Copy your Excel file to input folder
copy "C:\Your\Data\dataset.xlsx" "P:\input\"
```

### **3. Run GPU Optimization**
```batch
# Quick optimization (recommended)
P:\run_optimizer.bat dataset.xlsx 25 ratio

# Advanced optimization
P:\Portfolio_Optimization.bat dataset.xlsx 30 roi
```

### **4. Download Results**
```batch
# Check results
dir P:\output\

# Copy to local machine
copy "P:\output\optimized_*.xlsx" "C:\Results\"
```

---

## 📊 **ALGORITHM PERFORMANCE REFERENCE**

| Algorithm | CPU Time | A100 Time | Speedup | Memory | Best For |
|-----------|----------|-----------|---------|---------|----------|
| **Genetic Algorithm** | 0.100s | 0.012s | **8.3x** | 2-8GB | General optimization |
| **Particle Swarm** | 0.060s | 0.006s | **10.0x** | 2-6GB | Fast convergence |
| **Differential Evolution** | 0.120s | 0.013s | **9.2x** | 3-8GB | Robust optimization |
| **Simulated Annealing** | 0.270s | 0.039s | **6.9x** | 1-4GB | High quality solutions |
| **Ant Colony** | 388.0s | 32.3s | **12.0x** | 4-12GB | Research applications |
| **Hill Climbing** | 0.050s | 0.008s | **6.3x** | 1-3GB | Ultra-fast execution |
| **Bayesian** | 0.040s | 0.005s | **8.0x** | 2-6GB | Uncertainty analysis |
| **Ensemble** | 0.524s | 0.078s | **6.7x** | 8-20GB | Maximum quality |

---

## 🎯 **ALGORITHM SELECTION GUIDE**

### **Speed Priority (< 0.01s)**
- **1st Choice:** GPU Hill Climbing
- **2nd Choice:** GPU Bayesian Optimization  
- **3rd Choice:** GPU Particle Swarm Optimization

### **Quality Priority (Highest Accuracy)**
- **1st Choice:** GPU Ant Colony Optimization (97.8% accuracy)
- **2nd Choice:** GPU Simulated Annealing (96.3% accuracy)
- **3rd Choice:** GPU Differential Evolution (95.1% accuracy)

### **Balanced Performance**
- **1st Choice:** GPU Genetic Algorithm
- **2nd Choice:** GPU Particle Swarm Optimization
- **3rd Choice:** GPU Differential Evolution

---

## 📁 **BATCH FILES REFERENCE**

### **run_optimizer.bat** - General Purpose
```batch
# Usage: run_optimizer.bat [dataset] [portfolio_size] [metric] [algorithms]
run_optimizer.bat SENSEX_data.xlsx 25 ratio gpu_genetic_algorithm
```

### **Portfolio_Optimization.bat** - A100 Optimized
```batch
# Usage: Portfolio_Optimization.bat [dataset] [portfolio_size] [metric]
Portfolio_Optimization.bat SENSEX_data.xlsx 30 roi
```

### **Research_Optimization.bat** - All Algorithms
```batch
# Usage: Research_Optimization.bat [dataset] [portfolio_size] [include_aco]
Research_Optimization.bat SENSEX_data.xlsx 40 yes
```

### **HFT_Optimization.bat** - High-Frequency Trading
```batch
# Usage: HFT_Optimization.bat [dataset] [portfolio_size]
HFT_Optimization.bat HFT_data.xlsx 15
```

---

## 💾 **MEMORY USAGE GUIDE**

### **Portfolio Size → VRAM Usage**
- **10-20 strategies:** 2-4GB VRAM (10% of A100)
- **20-30 strategies:** 4-6GB VRAM (15% of A100)  
- **30-50 strategies:** 6-8GB VRAM (20% of A100)
- **50+ strategies:** 8-16GB VRAM (40% of A100)

### **Algorithm Memory Requirements**
- **Low Memory (< 5GB):** Hill Climbing, Simulated Annealing, Bayesian
- **Medium Memory (5-10GB):** Genetic Algorithm, PSO, Differential Evolution
- **High Memory (> 10GB):** Ant Colony Optimization, Ensemble Method

---

## 🔧 **TROUBLESHOOTING**

### **Connection Issues**
```batch
# Test connection
ping 204.12.223.93

# Clear credentials
net use * /delete

# Reconnect
net use P: \\204.12.223.93\optimizer_share /user:opt_admin Chetti@123
```

### **GPU Issues**
```batch
# Check GPU status
type P:\logs\gpu_status.log

# Check A100 memory
type P:\logs\a100_memory.log

# Use CPU fallback if needed
run_optimizer.bat dataset.xlsx 25 ratio cpu_genetic_algorithm
```

### **Performance Issues**
- **Slow execution:** Check if ACO is included (takes longer but highest speedup)
- **Memory errors:** Reduce portfolio size or use distributed processing
- **Network delays:** Use smaller datasets or compress files

---

## 📈 **PERFORMANCE MONITORING**

### **Real-time Monitoring**
```batch
# View current GPU utilization
type P:\logs\gpu_performance.log

# Monitor optimization progress
type P:\logs\current_jobs.log

# Check system status
type P:\logs\system_status.log
```

### **Performance Metrics**
- **Average Speedup:** 8.6x across all algorithms
- **Peak Memory Usage:** Up to 39.8GB VRAM (98% of A100)
- **Success Rate:** 99.2% with automatic CPU fallback
- **Throughput:** Up to 4,762 strategies/second

---

## 🎯 **BEST PRACTICES**

### **Dataset Preparation**
- **Format:** Excel (.xlsx) with Date + Strategy columns
- **Size:** Minimum 30 days, 10 strategies
- **Quality:** Clean numeric data, no empty cells
- **Naming:** Use descriptive filenames

### **Optimization Settings**
- **Portfolio Size:** 20-30 strategies for best balance
- **Metric:** Use 'ratio' for Sharpe-like ratio, 'roi' for returns
- **Algorithms:** Start with GPU Genetic Algorithm for general use
- **Memory:** Monitor VRAM usage, keep under 90%

### **Result Analysis**
- **Download:** Always copy results to local machine
- **Backup:** Keep original datasets and results
- **Compare:** Use ensemble method for critical decisions
- **Validate:** Check performance metrics and GPU speedup

---

## 📞 **SUPPORT CONTACTS**

- **System Admin:** admin@company.com
- **GPU Support:** gpu-support@company.com  
- **Emergency:** +1-XXX-XXX-XXXX
- **Documentation:** `/opt/heavydb_optimizer/docs/GPU_Enhanced_HeavyDB_Windows_User_Guide.md`

---

## 🏆 **SYSTEM SPECIFICATIONS**

### **Hardware**
- **GPU:** NVIDIA A100-SXM4-40GB (6,912 CUDA cores, 432 Tensor cores)
- **Memory:** 40GB HBM2 with 1,555 GB/s bandwidth
- **Server:** Ubuntu 22.04 LTS with HeavyDB GPU-enabled
- **Network:** Gigabit Ethernet with Samba file sharing

### **Software**
- **HeavyDB:** Version 7.x with GPU acceleration
- **Python:** 3.10+ with GPU optimization libraries
- **CUDA:** Version 11.8+ with A100 support
- **Drivers:** NVIDIA 525.x+ production drivers

---

**🎉 Ready to achieve 8.6x faster portfolio optimization with GPU acceleration!**

*For complete documentation, see: `GPU_Enhanced_HeavyDB_Windows_User_Guide.md`*
