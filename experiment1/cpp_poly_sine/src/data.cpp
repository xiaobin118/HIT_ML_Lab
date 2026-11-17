#include "data.hpp"

#include <iomanip>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

const double pi = std::acos(-1.0);

Data generate_data(int n, double noise_std, unsigned seed) {
    // TODO:
    // 1. 校验 n > 0
    // 2. 创建 std::mt19937
    // 3. 创建 [-1, 1] 上的均匀分布
    // 4. 创建标准正态分布
    // 5. 依次填充 x 和 y
    // 6. 返回 Data
    (void)n;
    (void)noise_std;
    (void)seed;
    if (n <= 0) {
        throw std::invalid_argument("n must be greater than 0");
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);
    std::normal_distribution<double> normal_dist(0.0, noise_std);
    // 5. 依次填充 x 和 y
    Data data;
    data.x.resize(n);
    data.y.resize(n);
    for (int i = 0; i < n; ++i) {
        data.x[i] = uniform_dist(rng);
        data.y[i] = std::sin(2 * pi * data.x[i]) + normal_dist(rng) * noise_std;
    }
    return data;

}

Data make_test_data(int num) {
    // TODO:
    // 1. 校验 num > 0
    // 2. x[i] = -1 + 2*i/(num-1)
    // 3. y[i] = sin(2*pi*x[i])
    // 4. 返回 Data
    (void)num;
    if (num < 2) {
        throw std::invalid_argument("num must be greater than 2");
    }
    Data test_data;
    test_data.x.resize(num);
    test_data.y.resize(num);
    for (int i = 0; i < num; ++i) {
        test_data.x[i] = -1.0 + 2.0 * i / (num - 1);
        test_data.y[i] = std::sin(2 * pi * test_data.x[i]);
    }
    return test_data;
}

void save_csv(const std::string& filename, const Data& data) {
    // TODO:
    // 1. 打开 ofstream
    // 2. 写表头 x,y
    // 3. 逐行写 data.x[i], data.y[i]
    // 4. 检查文件是否正常打开
    (void)filename;
    (void)data;
    std::ofstream ofs(filename);
    ofs << std::setprecision(15);

    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    ofs << "x,y\n";
    for (size_t i = 0; i < data.x.size(); ++i) {
        ofs << data.x[i] << "," << data.y[i] << "\n";
    }
    ofs.close();
}
