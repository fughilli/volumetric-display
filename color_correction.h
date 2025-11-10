#pragma once

#include <algorithm>
#include <array>

namespace util {

template <int N>
struct ColorCorrectorOptions {
  // Gamma correction values for each channel.
  float gamma[N]{};

  // Brightness values for each channel. These are given in mcd in the
  // datasheet.
  float brightness[N]{};
};

constexpr ColorCorrectorOptions<3> kColorCorrectorWs2812bOptions = {
    {2.8f, 2.8f, 2.8f},
    {(550.0f + 700.0f) / 2.0f, (1100.0f + 1400.0f) / 2.0f, (200.0f + 400.0f) / 2.0f},
};

template <int N>
class ColorCorrector {
 public:
  using Options = ColorCorrectorOptions<N>;

  ColorCorrector(Options options) : options_(options) {
    float min_brightness = *std::min_element(options_.brightness, options_.brightness + N);

    for (int i = 0; i < N; ++i) {
      std::array<uint8_t, 256>& table = color_table_[i];

      options_.brightness[i] = min_brightness / options_.brightness[i];
      for (int j = 0; j < 256; ++j) {
        table[j] = static_cast<uint8_t>(
            std::ceil(std::pow(j / 255.0f, options_.gamma[i]) * 255.0f * options.brightness[i]));
      }
    }
  }

  void CorrectInPlace(uint8_t* pixel) const {
    if (pixel == nullptr)
      return;

    for (int i = 0; i < N; ++i)
      pixel[i] = color_table_[i][pixel[i]];
  }

  void CorrectPixelsInPlace(uint8_t* pixel_buffer, int num_pixels) const {
    for (int i = 0; i < num_pixels; ++i) {
      CorrectInPlace(pixel_buffer + i * N);
    }
  }

 private:
  Options options_{};
  std::array<std::array<uint8_t, 256>, N> color_table_{};
};

template <int N>
class ReverseColorCorrector {
 public:
  using Options = ColorCorrectorOptions<N>;

  ReverseColorCorrector(Options options) : options_(options) {
    float min_brightness = *std::min_element(options_.brightness, options_.brightness + N);

    for (int i = 0; i < N; ++i) {
      std::array<uint8_t, 256>& table = reverse_color_table_[i];

      float scale = options_.brightness[i] / min_brightness;
      float inv_gamma = 1.0f / options_.gamma[i];

      for (int j = 0; j < 256; ++j) {
        float normalized = j / 255.0f / scale;  // Undo peak brightness scaling
        normalized = std::clamp(normalized, 0.0f, 1.0f);

        table[j] = static_cast<uint8_t>(std::ceil(std::pow(normalized, inv_gamma) * 255.0f));
      }
    }
  }

  void ReverseCorrectInPlace(uint8_t* pixel) const {
    if (pixel == nullptr)
      return;

    for (int i = 0; i < N; ++i) {
      pixel[i] = reverse_color_table_[i][pixel[i]];
    }
  }

  void ReverseCorrectPixelsInPlace(uint8_t* pixel_buffer, int num_pixels) const {
    for (int i = 0; i < num_pixels; ++i) {
      ReverseCorrectInPlace(pixel_buffer + i * N);
    }
  }

 private:
  Options options_{};
  std::array<std::array<uint8_t, 256>, N> reverse_color_table_{};
};

}  // namespace util
