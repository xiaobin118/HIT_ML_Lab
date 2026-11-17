#pragma once

#include <string>
#include <vector>

struct Data {
    std::vector<double> x;
    std::vector<double> y;
};

// 生成训练数据：x ~ U(-1, 1), y = sin(2*pi*x) + noise_std * N(0,1)
Data generate_data(int n, double noise_std, unsigned seed);

// 生成无噪声测试数据：x 在 [-1,1] 上等距，y = sin(2*pi*x)
Data make_test_data(int num);

// 把 x, y 写进 CSV，表头为 x,y
void save_csv(const std::string& filename, const Data& data);
