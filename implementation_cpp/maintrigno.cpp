#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <string> 

#include "proposed_trigno.h"
#include "taylorsimd.h"

using namespace std;
using namespace std::chrono;

//=============================================================================
// Benchmarking Infrastructure
//=============================================================================
using FuncPtr = double(*)(double);

struct BenchmarkResult {
    double angle;
    double std_result;
    double my_result;
    double abs_error;
    double rel_error; // When std_result is near zero, this is set to 0.
    long long std_time_ns; // Execution time for built-in function.
    long long my_time_ns;  // Execution time for your implementation.
};

struct Bench_Response {
    double abs_err_max;
    double abs_err_avg;
    double abs_std_dev;
    double rel_err_max;
    double rel_err_avg;
    double rel_std_dev;
    long long std_time_total;
    long long my_func_total;
    int wins;
    int losses;
    int equals;
};

void save_performance_data(const std::vector<BenchmarkResult>& results, const char* method_name) {
    std::string filename = std::string(method_name) + ".csv";  
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write header row
    file << "angle,std_result,my_result,abs_error,rel_error,std_time,my_time,diff,speedup_ratio\n";

    // Write data rows
    for (const auto& r : results) {
        long long diff = r.std_time_ns - r.my_time_ns;
        double speedup_ratio;

        if (r.std_time_ns == 0 && r.my_time_ns == 0) {
            speedup_ratio = 1.0;  // No difference in execution
        } else if (r.my_time_ns == 0) {
            speedup_ratio = std::numeric_limits<double>::infinity();  // My function is extremely fast
        } else if (r.std_time_ns == 0) {
            speedup_ratio = 0.0;  // My function is infinitely slower
        } else {
            speedup_ratio = static_cast<double>(r.std_time_ns) / r.my_time_ns;
        }

        file << r.angle << "," 
             << r.std_result << "," 
             << r.my_result << "," 
             << r.abs_error << "," 
             << r.rel_error << "," 
             << r.std_time_ns << "," 
             << r.my_time_ns << "," 
             << diff << ","
             << speedup_ratio << "\n";
    }
    
    file.close();
    std::cout << "Saved performance data for " << method_name << " to " << filename << std::endl;
}


template<typename Func>
long long measure_execution_time(Func f, double input, size_t iterations) {
    volatile double res;
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i)
        res = f(input);
    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count();
}

Bench_Response run_benchmark(const char* func_name, FuncPtr std_func, FuncPtr my_func,
                             size_t num_angles = 1000, size_t iterations = 100000) {
    vector<BenchmarkResult> results;
    results.reserve(num_angles);

    mt19937_64 rng(2024873);
    uniform_real_distribution<double> angle_dist(-1e6, 1e6);
    const double epsilon = 1e-12;

    for (size_t i = 0; i < num_angles; ++i) {
        double angle = angle_dist(rng);
        double std_val = std_func(angle);
        double my_val  = my_func(angle);
        double abs_err = fabs(my_val - std_val);
        double rel_err = (fabs(std_val) > epsilon) ? abs_err / fabs(std_val) : 0.0;
        long long std_time = measure_execution_time(std_func, angle, iterations);
        long long my_time  = measure_execution_time(my_func, angle, iterations);
        results.push_back({angle, std_val, my_val, abs_err, rel_err, std_time, my_time});
    }
    Bench_Response Local_Entry = {};
    double sum_abs = 0.0, sum_rel = 0.0;
    double max_abs = 0.0, max_rel = 0.0;
    vector<double> abs_errs, rel_errs;
    long long total_std_time = 0, total_my_time = 0;
    int losses = 0, wins = 0, equals = 0;
    
    for (const auto &r : results) {
        if (r.my_time_ns > r.std_time_ns){
            losses++;
        }else if (r.my_time_ns < r.std_time_ns){
            wins++;
        }else{
            equals++;
        }
        sum_abs += r.abs_error;
        sum_rel += r.rel_error;
        if (r.abs_error > max_abs)
            max_abs = r.abs_error;
        if (r.rel_error > max_rel)
            max_rel = r.rel_error;
        abs_errs.push_back(r.abs_error);
        rel_errs.push_back(r.rel_error);
        total_std_time += r.std_time_ns;
        total_my_time  += r.my_time_ns;
    }
    
    size_t result_size = results.size();
    double avg_abs = sum_abs / result_size;
    double avg_rel = sum_rel / result_size;
    double stddev_abs = 0.0, stddev_rel = 0.0;
    for (auto e : abs_errs)
        stddev_abs += (e - avg_abs) * (e - avg_abs);
    for (auto e : rel_errs)
        stddev_rel += (e - avg_rel) * (e - avg_rel);
    stddev_abs = sqrt(stddev_abs / result_size);
    stddev_rel = sqrt(stddev_rel / result_size);

    Local_Entry.abs_err_max = max_abs;
    Local_Entry.abs_err_avg = avg_abs;
    Local_Entry.abs_std_dev = stddev_abs;
    Local_Entry.rel_err_max = max_rel;
    Local_Entry.rel_err_avg = avg_rel;
    Local_Entry.rel_std_dev = stddev_rel;
    Local_Entry.std_time_total = total_std_time;
    Local_Entry.my_func_total = total_my_time;
    Local_Entry.wins = wins;
    Local_Entry.losses = losses;
    Local_Entry.equals = equals;

    // save_performance_data(results, func_name); // to save data to csv files

    return Local_Entry;
}

struct Final_Total_Stats {
    double total_abs_err_max = 0.0;
    double total_abs_err_avg = 0.0;
    double total_abs_std_dev = 0.0;
    double total_rel_err_max = 0.0;
    double total_rel_err_avg = 0.0;
    double total_rel_std_dev = 0.0;
    long long total_std_time_total = 0;
    long long total_my_func_total = 0;
    double total_wins = 0.0;
    double total_losses = 0.0;
    double total_equals = 0.0;
};

void updateStats(Final_Total_Stats& total_stats, const Bench_Response& new_data) {
    total_stats.total_abs_err_max += new_data.abs_err_max;
    total_stats.total_abs_err_avg += new_data.abs_err_avg;
    total_stats.total_abs_std_dev += new_data.abs_std_dev;
    total_stats.total_rel_err_max += new_data.rel_err_max;
    total_stats.total_rel_err_avg += new_data.rel_err_avg;
    total_stats.total_rel_std_dev += new_data.rel_std_dev;
    total_stats.total_std_time_total += new_data.std_time_total;
    total_stats.total_my_func_total += new_data.my_func_total;
    total_stats.total_wins += new_data.wins;
    total_stats.total_losses += new_data.losses;
    total_stats.total_equals += new_data.equals;
}

void printStats(const Final_Total_Stats& stats, const string& label) {
    cout << "==============================================" << endl;
    cout << "         AVERAGE " << label << " STATISTICS" << endl;
    cout << "==============================================" << endl;
    cout << fixed << setprecision(6);
    cout << left << setw(45) << "Absolute Max Error: " << stats.total_abs_err_max << endl;
    cout << left << setw(45) << "Average Absolute Error: " << stats.total_abs_err_avg << endl;
    cout << left << setw(45) << "Average Absolute Std Dev: " << stats.total_abs_std_dev << endl;
    cout << left << setw(45) << "Relative Max Error: " << stats.total_rel_err_max << endl;
    cout << left << setw(45) << "Average Relative Error: " << stats.total_rel_err_avg << endl;
    cout << left << setw(45) << "Average Relative Std Dev: " << stats.total_rel_std_dev << endl;
    cout << left << setw(45) << "Total Standard Time: " << stats.total_std_time_total << endl;
    cout << left << setw(45) << "Total My Function Time: " << stats.total_my_func_total << endl;
    cout << left << setw(45) << "Wins: " << stats.total_wins << endl;
    cout << left << setw(45) << "Losses: " << stats.total_losses << endl;
    cout << left << setw(45) << "Equals: " << stats.total_equals << endl;
    cout << "==============================================" << endl;
}

//=============================================================================
// Main: Run Benchmarks for sin, cos, and tan (and their Taylor variants)
//=============================================================================
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    Final_Total_Stats sin_stats;
    Final_Total_Stats cos_stats;
    Final_Total_Stats tan_stats;
    Final_Total_Stats taylor_sin5_stats;
    Final_Total_Stats taylor_cos5_stats;
    Final_Total_Stats taylor_tan5_stats;
    Final_Total_Stats taylor_sin7_stats;
    Final_Total_Stats taylor_cos7_stats;
    Final_Total_Stats taylor_tan7_stats;
    Final_Total_Stats taylor_sin9_stats;
    Final_Total_Stats taylor_cos9_stats;
    Final_Total_Stats taylor_tan9_stats;
    
    Bench_Response sin_bench = run_benchmark("proposed_sin", sin, proposed_sin);
    updateStats(sin_stats, sin_bench);
    Bench_Response cos_bench = run_benchmark("proposed_cos", cos, proposed_cos);
    updateStats(cos_stats, cos_bench);
    Bench_Response tan_bench = run_benchmark("proposed_tan", tan, proposed_tan);
    updateStats(tan_stats, tan_bench);
    Bench_Response taylor_sin5_bench = run_benchmark("taylorsin5", sin, taylorSin5);
    updateStats(taylor_sin5_stats, taylor_sin5_bench);
    Bench_Response taylor_cos5_bench = run_benchmark("taylorcos5", cos, taylorCos5);
    updateStats(taylor_cos5_stats, taylor_cos5_bench);
    Bench_Response taylor_tan5_bench = run_benchmark("taylortan5", tan, maclaurinTan5);
    updateStats(taylor_tan5_stats, taylor_tan5_bench);
    Bench_Response taylor_sin7_bench = run_benchmark("taylorsin7", sin, taylorSin7);
    updateStats(taylor_sin7_stats, taylor_sin7_bench);
    Bench_Response taylor_cos7_bench = run_benchmark("taylorcos7", cos, taylorCos7);
    updateStats(taylor_cos7_stats, taylor_cos7_bench);
    Bench_Response taylor_tan7_bench = run_benchmark("taylortan7", tan, maclaurinTan7);
    updateStats(taylor_tan7_stats, taylor_tan7_bench);
    Bench_Response taylor_sin9_bench = run_benchmark("taylorsin9", sin, taylorSin9);
    updateStats(taylor_sin9_stats, taylor_sin9_bench);
    Bench_Response taylor_cos9_bench = run_benchmark("taylorcos9", cos, taylorCos9);
    updateStats(taylor_cos9_stats, taylor_cos9_bench);
    Bench_Response taylor_tan9_bench = run_benchmark("taylortan9", tan, maclaurinTan9);
    updateStats(taylor_tan9_stats, taylor_tan9_bench);
    
     
    printStats(sin_stats, "SINE");
    printStats(cos_stats, "COSINE");
    printStats(tan_stats, "TANGENT");
    printStats(taylor_sin5_stats, "TAYLOR 5 SINE");
    printStats(taylor_cos5_stats, "TAYLOR 5 COSINE");
    printStats(taylor_tan5_stats, "TAYLOR 5 TANGENT");
    printStats(taylor_sin7_stats, "TAYLOR 7 SINE");
    printStats(taylor_cos7_stats, "TAYLOR 7 COSINE");
    printStats(taylor_tan7_stats, "TAYLOR 7 TANGENT");
    printStats(taylor_sin9_stats, "TAYLOR 9 SINE");
    printStats(taylor_cos9_stats, "TAYLOR 9 COSINE");
    printStats(taylor_tan9_stats, "TAYLOR 9 TANGENT");

    
    return 0;
}
