# Trigonometric SIMD Benchmark - C++ Implementation

Disclaimer : This code is obsolete. SOTA FastMath functions now exist at [hwy/contrib/math/fast_math-inl.h](https://github.com/google/highway/blob/master/hwy/contrib/math/fast_math-inl.h) 

This repository accompanies the research paper **"A SIMD-Optimized Approximate Implementation for Fast and Memory-Efficient Trigonometric Computation"** by **Nikhil Dev Goyal** and **Parth Arora**.

📄 **Preprint Paper**  
📌 **DOI:** [https://doi.org/10.48550/arXiv.2502.10831](https://doi.org/10.48550/arXiv.2502.10831)  
📌 **Full Preprint Paper:** [arXiv Link](https://arxiv.org/html/2502.10831v1)  

Note : Pre-print version might be outdated

The paper introduces a set of trigonometric functions that are significantly faster than standard C++ implementations, achieving up to a **5× speed increase**. These functions are also highly **memory-efficient**, requiring no precomputations, making them ideal for hardware implementations on **low-end FPGAs and MCUs**. Benchmark comparisons demonstrate substantial **hardware resource reductions**, including DSPs, LUTs, and flip-flops, when compared to built-in functions.

---

## 📥 Getting Started
Follow these steps to **set up, compile, and run the code** to reproduce the results.

---

## 🛠️ Building the Project (Using CMake)

### **1️⃣ Clone the repository**
```sh
git clone https://github.com/Nikhil0250/trigno_simd_cpp.git
cd trigno_simd_cpp
```

### **2️⃣ Create a Build Directory**
```sh
mkdir build
cd build
```

### **3️⃣ Run CMake**
#### ➤ **Auto-Detect MinGW (Recommended)**
```sh
cmake -G "MinGW Makefiles" ..
```
This will automatically detect `gcc`, `g++`, and `mingw32-make`.

#### ➤ **If Auto-Detection Fails**
Manually specify compilers:
```sh
cmake -G "MinGW Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
```
If MinGW is installed in a custom location:
```sh
cmake -G "MinGW Makefiles" -DCMAKE_MAKE_PROGRAM="C:/YourMinGW/bin/mingw32-make.exe" ..
```

### **4️⃣ Compile the Project**
```sh
cmake --build .
```
✅ This compiles the project and generates the executable in `bin/trigno_benchmark.exe`.

---

## 🔥 Running the Benchmark
After compilation, run:
```sh
bin\trigno_benchmark.exe
```
✅ This will generate the **benchmark results in CSV format** inside the `data/` folder.
✅ A detailed performance summary is displayed in the terminal upon execution.
---

## 📊 Reproducing Research Paper Results
The results of the benchmark will be saved in the `data/` directory:
```
data/
├── proposed_sin.csv
├── proposed_cos.csv
├── proposed_tan.csv
├── taylorcos5.csv
├── taylorsin5.csv
├── taylortan5.csv
├── ...
```
Each CSV contains:
- **`angle`**: Input angle in radians
- **`std_result`**: Standard C++ function output
- **`my_result`**: Optimized SIMD function output
- **`abs_error`**: Absolute error between standard and optimized functions
- **`rel_error`**: Relative error
- **`std_time`**: Execution time of standard function (ns)
- **`my_time`**: Execution time of optimized function (ns)
- **`speedup_ratio`**: Speed improvement compared to the standard function

---

## 📜 Project Structure
```
trigno_simd_cpp/
│── include/                      # Header files (.h)
│   ├── angle_reduction.h          # Header for angle reduction optimizations
│   ├── proposed_trigno.h          # Header for **proposed optimized trigonometric functions**
│   ├── taylorsimd.h               # Header for SIMD-optimized Taylor series functions
│
│── src/                          # Source files (.cpp)
│   ├── angle_reduction.cpp        # Implements angle reduction techniques
│   ├── benchmark.cpp              # Main benchmarking script 
│   ├── proposed_trigno.cpp        # Implements **proposed SIMD-optimized trigonometric functions**
│   ├── taylorsimd.cpp             # SIMD-optimized Taylor series calculations
│
│── data/                         # Computed benchmark results (CSV files)
│   ├── proposed_sin.csv           # Benchmark results for proposed sin()
│   ├── proposed_cos.csv           # Benchmark results for proposed cos()
│   ├── proposed_tan.csv           # Benchmark results for proposed tan()
│   ├── taylorcos5.csv             # Taylor series approximation (5 terms) for cos()
│   ├── taylorsin5.csv             # Taylor series approximation (5 terms) for sin()
│   ├── taylortan5.csv             # Taylor series approximation (5 terms) for tan()
│   ├── ...                        # Other benchmark results
│
│── vitis_hls_code/               # FPGA HLS implementations (for hardware benchmarks)
│   ├── proposed_cos_SIMD.cpp        # HLS implementation for SIMD-optimized Proposed cos()
│   ├── proposed_sin_SIMD.cpp        # HLS implementation for SIMD-optimized Proposed sin()
│   ├── proposed_tan_SIMD.cpp        # HLS implementation for SIMD-optimized Proposed tan()
│   ├── taylor_cos_SIMD.cpp        # Taylor series-based cos() for FPGA
│   ├── taylor_sin_SIMD.cpp        # Taylor series-based sin() for FPGA
│   ├── taylor_tan_SIMD.cpp        # Taylor series-based tan() for FPGA
│   ├── testbench_proposed_cos_SIMD.cpp  # Testbench for cos()
│   ├── testbench_proposed_sin_SIMD.cpp  # Testbench for sin()
│   ├── testbench_proposed_tan_SIMD.cpp  # Testbench for tan()
│   ├── testbench_taylor_cos_SIMD.cpp  # Testbench for Taylor cos()
│   ├── testbench_taylor_sin_SIMD.cpp  # Testbench for Taylor sin()
│   ├── testbench_taylor_tan_SIMD.cpp  # Testbench for Taylor tan()
│
│── vitis_hls_reports/            # Reports from FPGA synthesis & simulations
│   ├── report_proposed_cos_SIMD/    # Reports for proposed_cos_SIMD
│   ├── report_proposed_sin_SIMD/    # Reports for proposed_sin_SIMD
│   ├── report_proposed_tan_SIMD/    # Reports for proposed_tan_SIMD
│   ├── report_taylor_cos_SIMD/    # Reports for Taylor cos() SIMD
│   ├── report_taylor_sin_SIMD/    # Reports for Taylor sin() SIMD
│   ├── report_taylor_tan_SIMD/    # Reports for Taylor tan() SIMD
│   ├── report_cos_inbuilt/        # Baseline hardware results for standard cos()
│   ├── report_sin_inbuilt/        # Baseline hardware results for standard sin()
│   ├── report_tan_inbuilt/        # Baseline hardware results for standard tan()
│
│── bin/                          # Compiled executables (generated after build)
│   ├── trigno_benchmark.exe       # Final compiled executable
│
│── CMakeLists.txt                 # CMake build configuration
│── README.md                      # Project documentation
│── LICENSE                        # License file
│── .gitignore                     # Files to ignore in Git

```

---

## 📬 Support & Contribution
If you have questions or want to contribute, feel free to **open an issue** or **submit a pull request**.
