# OpenCV 图像处理项目

## 项目简介

这是一个基于 OpenCV 的 C++ 图像处理项目，实现了多种基础的图像处理操作。项目通过对苹果图片进行多种处理，包括颜色空间转换、滤波、特征提取、图像绘制和几何变换等，展示了 OpenCV 库的基本功能。

## 环境配置

### 系统要求
- Ubuntu 20.04 或更高版本
- C++17 兼容编译器
- CMake 3.10 或更高版本

### 依赖安装

1. **安装编译工具和基础依赖**
```bash
sudo apt update
sudo apt install build-essential cmake git
```

2. **安装 OpenCV 库**
```bash
sudo apt install libopencv-dev
```
或者从源码编译安装最新版本：
```bash
sudo apt install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
cmake -j$(nproc) --build . --config Release
sudo cmake --install .
```

## 构建和运行

### 构建项目

1. **克隆项目或创建项目结构**
```bash
mkdir -p opencv_project/src opencv_project/resources
# 将 main.cpp 放入 src/ 目录
# 将 Red_Apple.jpg 放入 resources/ 目录
```

2. **配置和编译**
```bash
cd opencv_project
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### 运行程序
```bash
./opencv_project
```

程序运行后会在项目根目录下的 `result/` 文件夹中生成所有处理后的图像文件。

## 功能实现说明

### 1. 图像颜色空间转换
- **灰度图转换** (`1.1-convertToGray.png`): 使用 `cv::cvtColor` 将 BGR 图像转换为灰度图
- **HSV 转换** (`1.2-convertToHSV.png`): 将 BGR 图像转换为 HSV 颜色空间

### 2. 图像滤波
- **均值滤波** (`2.1-blur.png`): 使用 5×5 核进行均值模糊
- **高斯滤波** (`2.2-GaussianBlur.png`): 使用 5×5 核和标准差为5的高斯模糊

### 3. 特征提取与处理
- **颜色区域提取和轮廓检测** (`3.1-FeatureExtraction.png`):
  - 在 HSV 空间中提取红色区域
  - 使用 Canny 边缘检测
  - 查找并绘制最大轮廓
  - 计算并显示轮廓面积
  - 绘制边界框

- **图像形态学处理** (`3.2-FeatureExtractionAndProcessing.png`):
  - 灰度化和掩膜应用
  - 自适应阈值二值化
  - 膨胀和腐蚀操作

- **漫水填充处理** (`3.3-FeatureExtractionAndProcessing.png`):
  - 在随机位置进行漫水填充
  - 使用固定范围模式

### 4. 图像绘制
- **基本图形绘制** (`4.1-draw.png`):
  - 绘制绿色矩形
  - 绘制紫色圆形
  - 添加黄色文字

- **轮廓绘制** (`4.2-drawContours.png`):
  - 检测绘制图形的轮廓
  - 使用红色线条绘制

- **边界框绘制** (`4.3-drawBoundingBox.png`):
  - 为检测到的轮廓绘制边界框

### 5. 图像几何变换
- **图像旋转** (`5.1-Rotation.png`):
  - 以图像中心为轴旋转35度
  - 使用仿射变换

- **图像裁剪** (`5.2-Cut.png`):
  - 裁剪图像的左上角四分之一区域

## 技术特点

1. **现代 C++ 特性**:
   - 使用 C++17 标准
   - 模板元编程确保函数参数类型安全
   - 完美转发和 lambda 表达式

2. **模块化设计**:
   - 通用图像处理函数 `load_and_store`
   - 可复用的掩膜生成函数
   - 分离的关注点：加载、处理、保存

3. **错误处理**:
   - 输入验证
   - 空轮廓检查
   - 资源管理

## 常见问题解决

### 1. CMake 找不到 OpenCV
```
错误：Could not find a package configuration file provided by "OpenCV"
```
**解决方案**:
在`CMakeLists.txt`中添加：
```cmake
find_package(OpenCV REQUIRED)
add_executable(opencv_project src/main.cpp)
target_include_directories(opencv_project PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_project PRIVATE ${OpenCV_LIBS})
```

### 2. 在MSVC下编译带有中文注释的源代码文件会警告
```
警告：The file contains a character that cannot be represented in the current code page (936).
Save the file in Unicode format to prevent data lossMSVC(C4819)
```
**解决方案**
在`CMakeLists.txt`中添加：
```cmake
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif()
```

### 2. 图像文件找不到
```
错误：Can't open image file
```
**解决方案**:
- 确保 `Red_Apple.jpg` 文件位于 `resources/` 目录
- 检查文件路径和权限
- 在 `CMakeLists.txt` 中加入
  ```cmake
  add_custom_command(
      TARGET opencv_project
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_LIST_DIR}/resources/Red_Apple.jpg" "$<TARGET_FILE_DIR:opencv_project>/"
  )
  ```

## 代码结构说明

### 主函数 (`main.cpp`)
1. **模板函数 `load_and_store`**: 通用图像处理流水线
2. **Lambda 表达式**: 各处理步骤的实现
3. **处理流程**:
   - 加载图像
   - 应用处理函数
   - 保存结果

### 关键算法
1. **颜色提取**: HSV 空间的 `inRange` 操作
2. **轮廓检测**: Canny 边缘检测 + `findContours`
3. **形态学操作**: 膨胀、腐蚀、开运算、闭运算
4. **几何变换**: 旋转矩阵计算和仿射变换

## 许可证

本项目仅用于教学目的，遵循 MIT 许可证。

## 联系方式

如有问题或建议，请联系项目维护者或将问题提交至 GitHub Issues。

---

**注意**: 运行前请确保 resources 目录中包含 Red_Apple.jpg 图片文件。所有输出图片将保存在 result 目录中，程序运行时会自动创建该目录。
