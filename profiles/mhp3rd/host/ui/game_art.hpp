#pragma once

#include "imgui.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

// The game's own 2D art, for the port's screens that stand in for the game's:
// textures read from the .TMH0.14 containers in the player's DATA.BIN and
// decoded with the renderer's texture decoder, and text drawn the way the game
// draws its own, from the same glyphs. Nothing of the game is in the program.
//
// A .TMH0.14 container is ".TMH0.14", a count and a zero word, then per image:
// a 16-byte header whose first word is the image's size in bytes; the texels'
// 16-byte header (their size including it, 1, the GE texture format, width
// and height as halfwords) and the texels, swizzled; and for the indexed
// formats the palette's 16-byte header (its size, 1, the GE palette format,
// the number of entries) and its entries. Several palettes of an image follow
// each other; the game points the GE at one of them.
namespace mhp3rd::ui::art {

struct TmhImage {
    std::uint32_t format{};                // GE texture format: 4 and 5 are 4- and 8-bit indexed
    std::uint16_t width{};
    std::uint16_t height{};
    std::span<const std::uint8_t> texels;  // swizzled
    std::uint32_t palette_format{};        // GE palette format: 3 is RGBA8888
    std::span<const std::uint8_t> palettes;
};

// The images of a container, or none when `file` is not one.
[[nodiscard]] std::vector<TmhImage> parse_tmh(std::span<const std::uint8_t> file);

// Decodes `image` with palette `palette` (counted in the image's own palette
// size) into RGBA8888, red in the low byte. Empty when it cannot.
[[nodiscard]] std::vector<std::uint32_t> decode(const TmhImage &image, std::uint32_t palette);

// Part of a picture, in the game's texels. An empty region is the whole.
struct Region {
    int x{};
    int y{};
    int width{};
    int height{};
};

// A texture as the interface draws it: enlarged `scale` times with
// nearest-neighbour sampling, as the game samples its 2D art, so it stays
// sharp when drawn at the window's size.
struct Sheet {
    ImTextureData *texture{};
    float width{};   // in the game's texels
    float height{};
    [[nodiscard]] explicit operator bool() const noexcept { return texture != nullptr; }
    [[nodiscard]] ImTextureRef ref() const { return texture->GetTexRef(); }
    // Texture coordinates of the game's texel (u, v).
    [[nodiscard]] ImVec2 uv(float u, float v) const { return {u / width, v / height}; }
};

// Uploads `rgba` (width x height) or its region `part`, enlarged `scale` times.
// The sheet lives for the rest of the run.
[[nodiscard]] Sheet make_sheet(const std::vector<std::uint32_t> &rgba, int width, int height, Region part, int scale);

// Image `image` of the container in DATA.BIN entry `entry`, as the game gets
// the entry (mods included), with palette `palette`, its part `part`.
[[nodiscard]] Sheet load_sheet(std::uint32_t entry, std::size_t image, std::uint32_t palette, Region part, int scale);

// How the game sets a line of text: each character's whole glyph cell (`u` by
// `v` of it) squeezed into `width` by `height`, and `step` further for the
// next. In the game's screen pixels.
struct TextStyle {
    float width{};
    float height{};
    float step{};
    float u{};
    float v{};
};
// The large text of the game's menus (NEW GAME, CONTINUE) and its small text
// (messages, button hints).
inline constexpr TextStyle kMenuText{17.0f, 17.0f, 18.0f, 19.0f, 21.0f};
inline constexpr TextStyle kSmallText{6.0f, 13.0f, 7.0f, 18.0f, 21.0f};

// The game's glyphs for the printable ASCII characters, in cells as the game
// makes them (fonts/game_font.hpp): each glyph centred in a 20x20 cell and
// drawn twice, the second time 31/64 of a pixel further left, in 16 levels.
class Text {
public:
    // Makes the cells on first use. False when no font could be loaded.
    bool ready();
    // Draws `text` from `at`, the top left of its first cell, with `scale`
    // window pixels to a game pixel.
    void draw(ImDrawList *draw, ImVec2 at, float scale, const TextStyle &style, std::string_view text,
              ImU32 color) const;
    [[nodiscard]] static float width(const TextStyle &style, std::string_view text) {
        return text.empty() ? 0.0f : style.step * static_cast<float>(text.size() - 1u) + style.width;
    }

private:
    Sheet cells_;
    bool tried_{};
};

} // namespace mhp3rd::ui::art
