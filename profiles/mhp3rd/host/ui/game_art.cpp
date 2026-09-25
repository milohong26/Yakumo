#include "ui/game_art.hpp"

#include "fonts/game_font.hpp"
#include "gpu/texture_decode.hpp"
#include "mods/mhp3rd_mods.hpp"

#include "imgui_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mhp3rd::ui::art {
namespace {

std::uint32_t le32(std::span<const std::uint8_t> b, std::size_t at) {
    return static_cast<std::uint32_t>(b[at] | (b[at + 1u] << 8u) | (b[at + 2u] << 16u) | (b[at + 3u] << 24u));
}
std::uint16_t le16(std::span<const std::uint8_t> b, std::size_t at) {
    return static_cast<std::uint16_t>(b[at] | (b[at + 1u] << 8u));
}

std::uint32_t bits_per_texel(std::uint32_t format) {
    switch (format) {
    case 0u:
    case 1u:
    case 2u: return 16u;
    case 3u: return 32u;
    case 4u: return 4u;
    case 5u: return 8u;
    default: return 0u;
    }
}

// The cells of the text atlas: printable ASCII, 16 to a row, spaced as the
// game spaces them in its own atlas (a cell's height plus two apart).
constexpr int kCellColumns = 16;
constexpr int kCellWidth = fonts::kCell;
constexpr int kCellPitch = fonts::kCell + 2;
constexpr char32_t kFirstCharacter = U' ';
constexpr char32_t kLastCharacter = U'~';

} // namespace

std::vector<TmhImage> parse_tmh(std::span<const std::uint8_t> file) {
    std::vector<TmhImage> images;
    if (file.size() < 16u || std::memcmp(file.data(), ".TMH0.14", 8u) != 0) return images;
    const std::uint32_t count = le32(file, 8u);
    std::size_t at = 16u;
    for (std::uint32_t i = 0; i < count && at + 48u <= file.size(); ++i) {
        const std::uint32_t total = le32(file, at);
        const std::size_t texels = at + 16u;
        const std::uint32_t texel_bytes = le32(file, texels);
        TmhImage image;
        image.format = le32(file, texels + 8u);
        image.width = le16(file, texels + 12u);
        image.height = le16(file, texels + 14u);
        if (total < 32u || texel_bytes < 16u || texels + texel_bytes > file.size()) return {};
        image.texels = file.subspan(texels + 16u, texel_bytes - 16u);
        if (image.format == 4u || image.format == 5u) {
            const std::size_t palette = texels + texel_bytes;
            if (palette + 16u > file.size()) return {};
            const std::uint32_t palette_bytes = le32(file, palette);
            if (palette_bytes < 16u || palette + palette_bytes > file.size()) return {};
            image.palette_format = le32(file, palette + 8u);
            image.palettes = file.subspan(palette + 16u, palette_bytes - 16u);
        }
        images.push_back(image);
        at += total;
    }
    return images;
}

std::vector<std::uint32_t> decode(const TmhImage &image, std::uint32_t palette) {
    const std::uint32_t bits = bits_per_texel(image.format);
    if (bits == 0u || image.width == 0u || image.height == 0u) return {};
    gpu::TextureSnapshot snapshot;
    gpu::TextureState &state = snapshot.texture;
    state.enabled = true;
    state.format = static_cast<gpu::TextureFormat>(image.format);
    state.width = image.width;
    state.height = image.height;
    state.buffer_width = image.width;
    state.swizzled = true;
    state.clut_format = image.palette_format;
    state.clut_mask = 0xFFu;
    snapshot.row_bytes = image.width * bits / 8u;
    const std::size_t texel_bytes = static_cast<std::size_t>(snapshot.row_bytes) * image.height;
    if (image.texels.size() < texel_bytes) return {};
    snapshot.texels.assign(image.texels.begin(), image.texels.begin() + static_cast<std::ptrdiff_t>(texel_bytes));
    if (image.format == 4u || image.format == 5u) {
        const std::size_t entry = image.palette_format == 3u ? 4u : 2u;
        const std::size_t size = (image.format == 4u ? 16u : 256u) * entry;
        const std::size_t start = static_cast<std::size_t>(palette) * size;
        if (start + size > image.palettes.size()) return {};
        snapshot.clut.assign(image.palettes.begin() + static_cast<std::ptrdiff_t>(start),
                             image.palettes.begin() + static_cast<std::ptrdiff_t>(start + size));
    }
    std::vector<std::uint32_t> rgba;
    if (!gpu::decode_snapshot(snapshot, rgba)) return {};
    return rgba;
}

Sheet make_sheet(const std::vector<std::uint32_t> &rgba, int width, int height, Region part, int scale) {
    if (rgba.size() < static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) return {};
    if (part.width <= 0 || part.height <= 0) part = {0, 0, width, height};
    if (part.x < 0 || part.y < 0 || part.x + part.width > width || part.y + part.height > height) return {};
    scale = std::clamp(scale, 1, 8);
    auto *texture = new ImTextureData();
    texture->Create(ImTextureFormat_RGBA32, part.width * scale, part.height * scale);
    auto *out = reinterpret_cast<std::uint32_t *>(texture->GetPixels());
    for (int y = 0; y < part.height * scale; ++y) {
        const std::uint32_t *row = rgba.data() + static_cast<std::size_t>(part.y + y / scale) * width + part.x;
        for (int x = 0; x < part.width * scale; ++x)
            out[static_cast<std::size_t>(y) * part.width * scale + x] = row[x / scale];
    }
    ImGui::RegisterUserTexture(texture);
    return {texture, static_cast<float>(part.width), static_cast<float>(part.height)};
}

Sheet load_sheet(std::uint32_t entry, std::size_t image, std::uint32_t palette, Region part, int scale) {
    const std::vector<std::uint8_t> file = mods::entry(entry);
    const std::vector<TmhImage> images = parse_tmh(file);
    if (image >= images.size()) return {};
    const std::vector<std::uint32_t> rgba = decode(images[image], palette);
    if (rgba.empty()) return {};
    return make_sheet(rgba, images[image].width, images[image].height, part, scale);
}

bool Text::ready() {
    if (cells_ || tried_) return static_cast<bool>(cells_);
    tried_ = true;
    if (!fonts::ready()) return false;
    const int count = static_cast<int>(kLastCharacter - kFirstCharacter) + 1;
    const int rows = (count + kCellColumns - 1) / kCellColumns;
    const int width = kCellColumns * kCellWidth;
    const int height = rows * kCellPitch;
    std::vector<std::uint8_t> coverage(static_cast<std::size_t>(width) * height, 0u);
    for (char32_t code = kFirstCharacter; code <= kLastCharacter; ++code) {
        const int index = static_cast<int>(code - kFirstCharacter);
        const int cell_x = index % kCellColumns * kCellWidth;
        const int cell_y = index / kCellColumns * kCellPitch;
        const fonts::GlyphMetrics metrics = fonts::metrics(code);
        if (!metrics.found || metrics.width <= 0) continue;
        // Half-width characters are centred by their bitmap's width, their
        // top at 1 + ascender - bitmap top; a second pass 31/64 of a pixel to
        // the left makes them bolder.
        const int x = (fonts::kCell - metrics.width) / 2;
        const int y = 1 + fonts::kAscender - metrics.top;
        const auto put = [&](const fonts::GlyphBitmap &glyph, int dx) {
            for (int row = 0; row < glyph.height; ++row) {
                for (int column = 0; column < glyph.width; ++column) {
                    const int px = x + dx + glyph.x + column;
                    const int py = y + glyph.y + row;
                    if (px < 0 || px >= fonts::kCell || py < 0 || py >= fonts::kCell) continue;
                    std::uint8_t &out = coverage[static_cast<std::size_t>(cell_y + py) * width + cell_x + px];
                    out = std::max(out, glyph.pixels[static_cast<std::size_t>(row) * glyph.width + column]);
                }
            }
        };
        put(fonts::render(code, 0.0f, 0.0f), 0);
        put(fonts::render(code, 33.0f / 64.0f, 0.0f), -1);
    }
    std::vector<std::uint32_t> rgba(coverage.size());
    for (std::size_t i = 0; i < coverage.size(); ++i) {
        // The game keeps glyphs in 4 bits.
        const std::uint32_t alpha = static_cast<std::uint32_t>(coverage[i] >> 4u) * 17u;
        rgba[i] = alpha << 24u | 0x00FFFFFFu;
    }
    cells_ = make_sheet(rgba, width, height, {}, 1);
    return static_cast<bool>(cells_);
}

void Text::draw(ImDrawList *draw, ImVec2 at, float scale, const TextStyle &style, std::string_view text,
                ImU32 color) const {
    if (!cells_) return;
    float x = at.x;
    for (const char c : text) {
        const auto code = static_cast<char32_t>(static_cast<unsigned char>(c));
        if (code > kFirstCharacter && code <= kLastCharacter) {
            const int index = static_cast<int>(code - kFirstCharacter);
            const auto u = static_cast<float>(index % kCellColumns * kCellWidth);
            const auto v = static_cast<float>(index / kCellColumns * kCellPitch);
            draw->AddImage(cells_.ref(), {x, at.y}, {x + style.width * scale, at.y + style.height * scale},
                           cells_.uv(u, v), cells_.uv(u + style.u, v + style.v), color);
        }
        x += style.step * scale;
    }
}

} // namespace mhp3rd::ui::art
