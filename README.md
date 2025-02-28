\# Trigonometric SIMD Benchmark - C++ Implementation

This repository accompanies the research paper **"A Novel SIMD-Optimized Implementation for Fast and Memory-Efficient Trigonometric Computation"** by **Nikhil Dev Goyal** and **Parth Arora**.

📄 **Reference Research Paper:**  
📌 **DOI:** [https://doi.org/10.48550/arXiv.2502.10831](https://doi.org/10.48550/arXiv.2502.10831)  
📌 **Full Paper:** [arXiv Link](https://arxiv.org/html/2502.10831v1)  

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
│── include/               # Header files (.h)
│── src/                   # Source files (.cpp)
│── data/                  # CSV results
│── vitis_hls_code/        # FPGA HLS code
│── vitis_hls_reports/     # Reports
│── CMakeLists.txt         # CMake configuration
│── README.md              # Documentation
```

---

## 📬 Support & Contribution
If you have questions or want to contribute, feel free to **open an issue** or **submit a pull request**.
