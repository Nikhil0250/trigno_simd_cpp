## Getting Started
Here's how you can run the code and reproduce the results achieved in the research paper.
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nikhil0250/trigno_simd_cpp.git
2. **Navigate to implementation_cpp**
    ```bash
     cd implementation_cpp
3. **Building the code**
    ```bash
      g++ -std=c++17 -mavx2 -mfma -O3 -march=native maintrigno.cpp mytrigno.cpp taylorsimd.cpp angle_reduction.cpp -o output.exe
4. ** Running the code **
    ```bash
      ./output.exe  # Note: Use ./output.exe (not .\output.exe) on Linux/macOS  
   


