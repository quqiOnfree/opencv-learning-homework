#include <algorithm>
#include <format>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/**
 * @brief 加载图片并调用传入的特定格式的函数,并输出新图片
 * @param func 需要调用的函数
 * @param output_name 输出的照片名字
 */
template <
    class Func,
    std::enable_if_t<std::is_invocable_r_v<void, std::remove_cvref_t<Func>,
                                           cv::InputArray, cv::OutputArray>,
                     int> = 0>
void load_and_store(Func &&func, const std::string &output_name) {
  static cv::Mat img{cv::imread("Red_Apple.jpg")};
  cv::Mat out;
  std::invoke(std::forward<Func>(func), img, out);
  cv::imwrite(std::format("{}.png", output_name), out);
}

int main() {
  load_and_store(
      [](cv::InputArray src, cv::OutputArray dst) {
        // 转换图片颜色为灰色(单通道)
        cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
      },
      "1.1-convertToGray");

  load_and_store(
      [](cv::InputArray src, cv::OutputArray dst) {
        // 转换图片HSV
        cv::cvtColor(src, dst, cv::COLOR_BGR2HSV);
      },
      "1.2-convertToHSV");

  load_and_store(
      [](cv::InputArray src, cv::OutputArray dst) {
        // 均值滤波
        cv::blur(src, dst, {5, 5});
      },
      "2.1-blur");

  load_and_store(
      [](cv::InputArray src, cv::OutputArray dst) {
        // 高斯滤波
        cv::GaussianBlur(src, dst, {5, 5}, 5);
      },
      "2.2-GaussianBlur");

  /**
   * @brief 生成原图的掩膜,提取颜色
   * @param src 原图
   * @param dst 目标掩膜
   */
  auto make_mask = [](cv::InputArray src, cv::OutputArray dst) {
    cv::Mat blur, hsvd;
    cv::GaussianBlur(src, blur, {5, 5}, 5, 5);
    cv::cvtColor(blur, hsvd, cv::COLOR_BGR2HSV);
    cv::inRange(hsvd, cv::Scalar{0, 56, 0}, cv::Scalar{189, 255, 255}, dst);
  };

  load_and_store(
      [make_mask](cv::InputArray src, cv::OutputArray dst) {
        cv::Mat ranged, canny, dilated;
        make_mask(src, ranged);
        // 根据掩膜生成轮廓
        cv::Canny(ranged, canny, 25, 75);
        // 膨胀轮廓
        auto kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
        cv::dilate(canny, dilated, kernel);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        // 查找轮廓
        cv::findContours(dilated, contours, hierarchy, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);
        cv::Mat draw_contours{src.getMat().clone()};
        // 画出轮廓
        cv::drawContours(draw_contours, contours, -1, cv::Scalar{0, 255, 0},
                         10);
        // 查找最大的图形轮廓
        auto max_it =
            std::max_element(contours.begin(), contours.end(),
                             [](const std::vector<cv::Point> &a,
                                const std::vector<cv::Point> &b) {
                               return cv::contourArea(a) < cv::contourArea(b);
                             });
        if (max_it == contours.end()) {
          return;
        }
        // 获取轮廓矩形
        cv::Rect box = cv::boundingRect(*max_it);
        cv::rectangle(draw_contours, box, {255, 0, 255}, 10);
        // 画面积数值
        cv::putText(draw_contours,
                    std::format("S: {:.2f}", cv::contourArea(*max_it)),
                    {0, 200}, cv::FONT_HERSHEY_PLAIN, 10, {0, 0, 255}, 10);
        dst.getMatRef() = std::move(draw_contours);
      },
      "3.1-FeatureExtraction");

  load_and_store(
      [make_mask](cv::InputArray src, cv::OutputArray dst) {
        cv::Mat gray, ranged, bit_and, thrh;
        make_mask(src, ranged);
        // 灰度化
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        // 从掩膜获取图像
        cv::bitwise_and(gray, gray, bit_and, ranged);
        // 自适应二值化
        cv::adaptiveThreshold(bit_and, thrh, 255,
                              cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                              55, 0);
        auto kernel = cv::getStructuringElement(cv::MORPH_RECT, {5, 5});
        // 膨胀与腐蚀（闭运算）
        cv::morphologyEx(thrh, dst, cv::MORPH_CLOSE, kernel);
      },
      "3.2-FeatureExtractionAndProcessing");

  load_and_store(
      [make_mask](cv::InputArray src, cv::OutputArray dst) {
        cv::Mat gray, ranged, bit_and, thrh, dil, ero;
        make_mask(src, ranged);
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        cv::bitwise_and(gray, gray, bit_and, ranged);
        cv::adaptiveThreshold(bit_and, thrh, 255,
                              cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                              55, 0);
        auto kernel = cv::getStructuringElement(cv::MORPH_RECT, {5, 5});
        cv::dilate(thrh, dil, kernel);
        cv::erode(dil, ero, kernel);

        // 漫水处理
        cv::RNG rng{std::random_device{}()};
        int connectivity = 4;
        int maskVal = 255;
        int flags = connectivity | (maskVal << 8) | cv::FLOODFILL_FIXED_RANGE;
        cv::Scalar lowDiff{20, 20, 20}, upDiff{20, 20, 20};
        cv::Mat mask{cv::Mat::zeros(src.rows() + 2, src.cols() + 2, CV_8UC1)};
        cv::Mat img{src.getMat().clone()};
        for (std::size_t i = 0; i < 1000; ++i) {
          int px = rng.uniform(0, src.cols() - 1);
          int py = rng.uniform(0, src.rows() - 1);
          cv::Scalar color{static_cast<double>(rng.uniform(0, 255)),
                           static_cast<double>(rng.uniform(0, 255)),
                           static_cast<double>(rng.uniform(0, 255))};
          cv::Rect rect{};

          cv::floodFill(img, mask, {px, py}, color, &rect, lowDiff, upDiff,
                        flags);
        }
        dst.getMatRef() = std::move(img);
      },
      "3.3-FeatureExtractionAndProcessing");

  {
    // 生成新图片
    auto create_img = [](cv::OutputArray dst) {
      auto &img = dst.getMatRef();
      img = cv::Mat{cv::Size{800, 800}, CV_8UC3, {255, 255, 255}};
      cv::rectangle(img, {10, 20, 300, 200}, {0, 255, 0}, cv::FILLED);
      cv::circle(img, {600, 400}, 100, {255, 0, 255}, cv::FILLED);
      cv::putText(img, "Hello image!", {50, 700}, cv::FONT_HERSHEY_SIMPLEX, 2,
                  {255, 255, 0}, 10);
    };

    cv::Mat img;
    create_img(img);

    load_and_store([&img](cv::InputArray src,
                          cv::OutputArray dst) { dst.getMatRef() = img; },
                   "4.1-draw");

    // 获取轮廓
    auto get_contours = [](cv::InputArray src,
                           std::vector<std::vector<cv::Point>> &contours,
                           std::vector<cv::Vec4i> &hierarchy) {
      cv::Mat blur, gray, can, dil;
      cv::GaussianBlur(src, blur, {5, 5}, 5, 5);
      cv::cvtColor(blur, gray, cv::COLOR_BGR2GRAY);
      cv::Canny(gray, can, 50, 75);
      auto kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
      cv::dilate(can, dil, kernel);
      cv::findContours(dil, contours, hierarchy, cv::RETR_EXTERNAL,
                       cv::CHAIN_APPROX_SIMPLE);
    };

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    get_contours(img, contours, hierarchy);

    load_and_store(
        [&img, &contours](cv::InputArray src, cv::OutputArray dst) {
          auto &local_img = dst.getMatRef();
          local_img = img.clone();
          // 画出轮廓
          cv::drawContours(local_img, contours, -1, {0, 0, 255}, 3);
        },
        "4.2-drawContours");

    load_and_store(
        [&img, &contours](cv::InputArray src, cv::OutputArray dst) {
          auto &local_img = dst.getMatRef();
          local_img = img.clone();
          // 画出bounding box
          for (auto &obj : std::as_const(contours)) {
            auto rect = cv::boundingRect(obj);
            cv::rectangle(local_img, rect, {0, 0, 255}, 3);
          }
        },
        "4.3-drawBoundingBox");
  }

  load_and_store(
      [](cv::InputArray src, cv::OutputArray dst) {
        // 旋转图像
        auto matrix = cv::getRotationMatrix2D(
            {src.cols() / 2.0f, src.rows() / 2.0f}, 35, 1);
        cv::warpAffine(src, dst, matrix, {});
      },
      "5.1-Rotation");

  load_and_store(
      [](cv::InputArray src, cv::OutputArray dst) {
        // 裁剪图像
        dst.getMatRef() =
            src.getMat()({0, src.rows() / 2}, {0, src.cols() / 2});
      },
      "5.2-Cut");
}
