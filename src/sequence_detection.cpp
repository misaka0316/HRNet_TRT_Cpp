#include <vector>
#include <iostream>
#include <cmath> 
#include <optional>

std::optional<std::pair<std::vector<double>, int>> find_decreasing_segment(
    const std::vector<double>& angles, 
    double threshold, 
    int min_length = 5, 
    int max_length = 20
) {
    int n = angles.size();

    // 遍历开始点
    for (int start = 0; start < n; ++start) {
        if (angles[start] > 160) {
            if (n - start < min_length) {
                return std::nullopt; // 剩余长度不足，直接返回
            }

            // 遍历结束点
            for (int end = start; end < n; ++end) {
                for (int end1 = end + 2; end1 < n; ++end1) {
                    // 提取子序列
                    std::vector<double> segment(angles.begin() + end, angles.begin() + end1 + 1);

                    // 检查子序列是否递减
                    bool is_decreasing = true;
                    for (size_t i = 0; i < segment.size() - 1; ++i) {
                        double diff = segment[i] - segment[i + 1];
                        if (!(segment[i] > segment[i + 1] && diff >= 5 && diff <= 90)) {
                            is_decreasing = false;
                            break;
                        }
                    }

                    // 如果满足条件，返回子序列和索引
                    if (is_decreasing && segment.back() <= threshold && segment.back() > 10 &&
                        segment.size() >= min_length && segment.size() <= max_length) {
                        return std::make_pair(segment, end1);
                    } else if (!is_decreasing) {
                        break; // 退出当前循环
                    }
                }
            }
        }
    }

    return std::nullopt; // 未找到符合条件的子序列
}

std::optional<std::vector<double>> find_increasing_segment_after_decreasing(
    const std::vector<double>& angles,
    int decreasing_segment_end_index,
    double threshold,
    int min_length = 5,
    int max_length = 20
) {
    int n = angles.size();

    // 检查是否有足够的点用于递增段
    if (n - decreasing_segment_end_index < 3) {
        return std::nullopt;
    }

    // 从递减子序列的下一个点到之后的第5个点作为可能的起始点
    for (int start = decreasing_segment_end_index; start < decreasing_segment_end_index + 6 && start < n; ++start) {

        for (int end = start + 2; end < n; ++end) { // 保证至少3帧长的子序列
            // 提取子序列
            std::vector<double> segment(angles.begin() + start, angles.begin() + end + 1);

            // 检查子序列是否递增且符合角度差条件
            bool is_increasing = true;
            for (size_t i = 0; i < segment.size() - 1; ++i) {
                double diff = segment[i + 1] - segment[i];
                if (!(segment[i] < segment[i + 1] && diff >= 5 && diff <= 90)) {
                    is_increasing = false;
                    break;
                }
            }

            // 如果满足递增条件，检查其他要求
            if (is_increasing && segment.back() <= threshold && segment.back() >= 160 &&
                segment.size() >= min_length && segment.size() <= max_length) {
                return segment; // 返回符合条件的子序列
            } else if (!is_increasing) {
                break; // 如果递增条件失败，直接退出
            }
        }
    }

    return std::nullopt; // 如果没有找到符合条件的序列，返回空值
}


// 示例主函数
int main() {
    // std::vector<double> angles = {180, 170, 160, 120, 90, 50, 40, 30, 20};
    // double threshold = 50;

    // auto result = find_decreasing_segment(angles, threshold);
    // if (result.has_value()) {
    //     auto [segment, end_index] = result.value();
    //     std::cout << "Found segment: ";
    //     for (double angle : segment) {
    //         std::cout << angle << " ";
    //     }
    //     std::cout << "\nEnd index: " << end_index << std::endl;
    // } else {
    //     std::cout << "No segment found." << std::endl;
    // }

    std::vector<double> angles = {30, 40, 50, 60, 160, 170, 175, 180, 120, 110};
    int decreasing_segment_end_index = 4;
    double threshold = 160;

    auto result = find_increasing_segment_after_decreasing(angles, decreasing_segment_end_index, threshold);

    if (result.has_value()) {
        const auto& segment = result.value();
        std::cout << "Found segment: ";
        for (double angle : segment) {
            std::cout << angle << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "No segment found." << std::endl;
    }

    return 0;
}
