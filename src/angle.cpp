
// #include <iostream>
// #include <cmath> // for atan2 and degrees conversion

// using namespace std;

// struct Point {
//     double x, y;
// };

// double calculateAngle(Point A, Point B, Point C) {
//     // 创建向量 BA 和 BC
//     Point BA = {A.x - B.x, A.y - B.y};
//     Point BC = {C.x - B.x, C.y - B.y};

//     // 计算向量的弧度制角度
//     double angleBA = atan2(BA.y, BA.x);
//     double angleBC = atan2(BC.y, BC.x);

//     // 计算两个角度之间的差异，并转换为角度制
//     double angleDifference = (angleBC - angleBA) * (180.0 / M_PI);

//     // 确保角度在 [0, 360] 度之间
//     if (angleDifference < 0)
//         angleDifference += 360;

//     // 返回较小的角度
//     return min(angleDifference, 360 - angleDifference);
// }

// int main() {
//     // 示例：定义三个点
//     Point A = {0, 1};
//     Point B = {0, 0}; // 顶点
//     Point C = {1, -1};

//     // 调用函数并打印结果
//     cout << "The angle formed by the three points is: " 
//          << calculateAngle(A, B, C) << " degrees" << endl;

//     return 0;
// }


#include <iostream>
#include <cmath> // for acos and degrees conversion

using namespace std;

struct Point {
    double x, y;
};

double dotProduct(Point A, Point B) {
    return A.x * B.x + A.y * B.y;
}

double magnitude(Point P) {
    return sqrt(P.x * P.x + P.y * P.y);
}

double calculateAngle(Point A, Point B, Point C) {
    // 创建向量 BA 和 BC
    Point BA = {A.x - B.x, A.y - B.y};
    Point BC = {C.x - B.x, C.y - B.y};

    // 计算点积和模长
    double dotProd = dotProduct(BA, BC);
    double magBA = magnitude(BA);
    double magBC = magnitude(BC);

    // 防止浮点数运算造成的除零错误
    if (magBA == 0 || magBC == 0) {
        cout << "One of the vectors is a zero vector." << endl;
        return 0;
    }

    // 计算 cos(theta)
    double cosTheta = dotProd / (magBA * magBC);

    // 纠正可能的浮点数精度问题
    if (cosTheta > 1.0) cosTheta = 1.0;
    if (cosTheta < -1.0) cosTheta = -1.0;

    // 计算弧度制角度并转换为角度制
    double angleRadians = acos(cosTheta);
    double angleDegrees = angleRadians * (180.0 / M_PI);

    return angleDegrees;
}

int main() {
    // 示例：定义三个点
    Point A = {0, 1};
    Point B = {0, 0}; // 顶点
    Point C = {1, 0};

    // 调用函数并打印结果
    cout << "The angle formed by the three points is: " 
         << calculateAngle(A, B, C) << " degrees" << endl;

    return 0;
}