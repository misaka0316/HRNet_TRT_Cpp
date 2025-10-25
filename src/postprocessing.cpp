// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <random>
// #include <omp.h> // OpenMP 库用于并行化
// #include <chrono> // 用于记录时间

// // 模板函数声明
// template<typename T>
// void processHeatmaps(T* inputArray, std::vector<std::vector<int>>& result1, std::vector<T>& result2);

// template<typename T>
// void initializeRandomHeatmaps(T* arr, int numHeatmaps, int rows, int cols);

// int main() {
//     const int numHeatmaps = 17; // 热点图的数量
//     const int rows = 64;        // 每个热点图的行数
//     const int cols = 48;        // 每个热点图的列数
    
//     // 创建17个64x48的随机热点图，使用int类型（可以根据需要调整为float或其他类型）
//     float arr[numHeatmaps][rows][cols];
    
//     // 随机初始化热点图
//     initializeRandomHeatmaps<float>(&arr[0][0][0], numHeatmaps, rows, cols);
    
//     // 结果容器
//     std::vector<std::vector<int>> result1;  // 用于存储每个热点图最大值的二维坐标（行，列）
//     std::vector<float> result2;  // 用于存储每个热点图的最大值

//     // 记录程序开始时间
//     auto start = std::chrono::high_resolution_clock::now();
    
//     // 调用模板函数处理热点图
//     processHeatmaps<float>(&arr[0][0][0], result1, result2);
    
//     // 记录程序结束时间
//     auto end = std::chrono::high_resolution_clock::now();
    
//     // 计算时间差，单位为毫秒
//     std::chrono::duration<double> duration = end - start;
//     std::cout << "Time taken to process heatmaps: " << duration.count() << " seconds\n";
    
//     // 输出第一个结果：每个最大值的二维坐标（行，列）
//     std::cout << "Max value coordinates per heatmap:\n";
//     for (const auto& coords : result1) {
//         std::cout << "[" << coords[0] << ", " << coords[1] << "]\n";
//     }
    
//     // 输出第二个结果：每个热点图的最大值
//     std::cout << "Max values per heatmap:\n";
//     for (const auto& val : result2) {
//         std::cout << val << "\n";
//     }
    
//     return 0;
// }

// // 模板函数实现
// template<typename T>
// void processHeatmaps(T* inputArray, std::vector<std::vector<int>>& result1, std::vector<T>& result2) {
//     const int numHeatmaps = 17;
//     const int rows = 64;
//     const int cols = 48;
    
//     // 使用 OpenMP 进行并行化处理
//     #pragma omp parallel for
//     for (int i = 0; i < numHeatmaps; ++i) {
//         T maxVal = std::numeric_limits<T>::lowest(); // 使用类型T的最小值
//         int maxRow = -1, maxCol = -1;
        
//         // 计算当前热点图的起始地址，避免每次计算时重复计算
//         T* heatmap = inputArray + i * rows * cols;
        
//         // 遍历热点图（64x48），寻找最大值
//         for (int r = 0; r < rows; ++r) {
//             for (int c = 0; c < cols; ++c) {
//                 T value = *(heatmap + r * cols + c);
                
//                 // 更新最大值及其坐标
//                 if (value > maxVal) {
//                     maxVal = value;
//                     maxRow = r;
//                     maxCol = c;
//                 }
//             }
//         }
        
//         // 存储结果：最大值坐标
//         #pragma omp critical
//         {
//             result1.push_back({maxRow, maxCol});
//             result2.push_back(maxVal);
//         }
//     }
// }

// // 随机初始化热点图
// template <typename T>
// void initializeRandomHeatmaps(T* data, int width, int height, int channels) {
//     std::random_device rd;
//     std::mt19937 gen(rd());

//     if constexpr (std::is_integral_v<T>) {
//         std::uniform_int_distribution<T> dis(0, 100);  // 假设整数范围为0到100
//         for (int c = 0; c < channels; ++c) {
//             for (int y = 0; y < height; ++y) {
//                 for (int x = 0; x < width; ++x) {
//                     data[c * width * height + y * width + x] = dis(gen);
//                 }
//             }
//         }
//     } else if constexpr (std::is_floating_point_v<T>) {
//         std::uniform_real_distribution<T> dis(0.0, 100.0);  // 浮点数范围为0.0到1.0
//         for (int c = 0; c < channels; ++c) {
//             for (int y = 0; y < height; ++y) {
//                 for (int x = 0; x < width; ++x) {
//                     data[c * width * height + y * width + x] = dis(gen);
//                 }
//             }
//         }
//     } else {
//         static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "Template parameter must be an int or float point type");
//     }
// }

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits>
#include <omp.h> // OpenMP 用于并行化处理

// processHeatmaps 函数：处理每个热点图，找到最大值的坐标和最大值
template<typename T>
std::vector<std::vector<int>> processHeatmaps(T* inputArray, std::vector<T>& result2, int numHeatmaps, int rows, int cols) {
    std::vector<std::vector<int>> coords(numHeatmaps, std::vector<int>(2));

    #pragma omp parallel for
    for (int i = 0; i < numHeatmaps; ++i) {
        T maxVal = std::numeric_limits<T>::lowest();
        int maxRow = -1, maxCol = -1;

        T* heatmap = inputArray + i * rows * cols; // 计算当前热点图的起始地址

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                T value = *(heatmap + r * cols + c);
                if (value > maxVal) {
                    maxVal = value;
                    maxRow = r;
                    maxCol = c;
                }
            }
        }

        // 存储结果
        #pragma omp critical
        {
            coords[i][0] = maxCol; // X坐标
            coords[i][1] = maxRow; // Y坐标
            result2.push_back(maxVal);
        }
    }
    return coords;
}

// updateCoordinates 函数：根据热图的最大值坐标更新坐标
void updateCoordinates(std::vector<std::vector<int>>& coords, 
                       const std::vector<std::vector<std::vector<int>>>& batch_heatmaps, 
                       int heatmap_width, int heatmap_height) {
    const int numHeatmaps = coords.size();

    for (int n = 0; n < numHeatmaps; ++n) {
        int px = static_cast<int>(std::floor(coords[n][0] + 0.5));  
        int py = static_cast<int>(std::floor(coords[n][1] + 0.5));  

        if (1 < px && px < heatmap_width - 1 && 1 < py && py < heatmap_height - 1) {
            int diff_x = batch_heatmaps[n][py][px + 1] - batch_heatmaps[n][py][px - 1];
            int diff_y = batch_heatmaps[n][py + 1][px] - batch_heatmaps[n][py - 1][px];

            coords[n][0] += (diff_x > 0 ? 0.25 : (diff_x < 0 ? -0.25 : 0));
            coords[n][1] += (diff_y > 0 ? 0.25 : (diff_y < 0 ? -0.25 : 0));
        }
    }
}

// 随机初始化热点图
template<typename T>
void initializeRandomHeatmaps(T* arr, int numHeatmaps, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dis(0, 255); 

    for (int i = 0; i < numHeatmaps; ++i) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                *(arr + i * rows * cols + r * cols + c) = dis(gen); 
            }
        }
    }
}

// // 随机初始化热点图
// template <typename T>
// void initializeRandomHeatmaps(T* data, int width, int height, int channels) {
//     std::random_device rd;
//     std::mt19937 gen(rd());

//     if constexpr (std::is_integral_v<T>) {
//         std::uniform_int_distribution<T> dis(0, 100);  // 假设整数范围为0到100
//         for (int c = 0; c < channels; ++c) {
//             for (int y = 0; y < height; ++y) {
//                 for (int x = 0; x < width; ++x) {
//                     data[c * width * height + y * width + x] = dis(gen);
//                 }
//             }
//         }
//     } else if constexpr (std::is_floating_point_v<T>) {
//         std::uniform_real_distribution<T> dis(0.0, 100.0);  // 浮点数范围为0.0到1.0
//         for (int c = 0; c < channels; ++c) {
//             for (int y = 0; y < height; ++y) {
//                 for (int x = 0; x < width; ++x) {
//                     data[c * width * height + y * width + x] = dis(gen);
//                 }
//             }
//         }
//     } else {
//         static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "Template parameter must be an int or float point type");
//     }
// }

int main() {
    const int numHeatmaps = 17; 
    const int heatmap_width = 64;  
    const int heatmap_height = 48; 

    // 创建3D数组 batch_heatmaps，用于存储热图数据
    std::vector<std::vector<std::vector<int>>> batch_heatmaps(numHeatmaps, std::vector<std::vector<int>>(heatmap_height, std::vector<int>(heatmap_width)));

    // 随机初始化 batch_heatmaps 数据
    for (int i = 0; i < numHeatmaps; ++i) {
        for (int r = 0; r < heatmap_height; ++r) {
            for (int c = 0; c < heatmap_width; ++c) {
                // batch_heatmaps[i][r][c] = rand() % 256;
                batch_heatmaps[i][r][c] = rand();
            }
        }
    }

    // 创建存储最大值的结果容器
    std::vector<int> result2; // 存储最大值

    // 记录程序开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 处理热点图，获取最大值坐标和最大值
    auto coords = processHeatmaps(&batch_heatmaps[0][0][0], result2, numHeatmaps, heatmap_height, heatmap_width);

    // 调用 updateCoordinates 函数更新坐标
    updateCoordinates(coords, batch_heatmaps, heatmap_width, heatmap_height);

    // 记录程序结束时间
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to process heatmaps and update coordinates: " << duration.count() << " seconds\n";

    // 输出第一个结果：最大值坐标
    std::cout << "Max value coordinates per heatmap:\n";
    for (const auto& coords_pair : coords) {
        std::cout << "[" << coords_pair[0] << ", " << coords_pair[1] << "]\n";
    }

    // 输出第二个结果：最大值
    std::cout << "Max values per heatmap:\n";
    for (const auto& val : result2) {
        std::cout << val << "\n";
    }

    return 0;
}
