// The launcher: the screen Yakumo opens on, before the game starts. It is
// made to be taken for one of the game's own menus: it is built the way the
// game builds its GAME MENU, from the same textures (read from the player's
// DATA.BIN, never kept), the same sprite layout (traced from the running game
// with MHP3RD_TRACE_SPRITES) and the same text, drawn from the game's glyphs
// as the game draws them. The disc's menu music (PSP_GAME/SND0.AT3) plays
// while it is up. Its choices lead into the game and to Yakumo's settings,
// texture pack, mods and saves.
//
// The layout is in the game's 480x272 screen pixels. The frame's corners keep
// to the window's corners and the parchment and the bar at the bottom fill it,
// whatever its shape; the rest stays in the middle, in the PSP's proportions,
// as the game's own 2D does under Fill.

#include "ui/ui.hpp"

#include "ui/game_art.hpp"
#include "ui/layer.hpp"
#include "ui/menu.hpp"
#include "ui/widgets.hpp"

#include "audio/atrac_decoder.hpp"
#include "audio/audio_sink.hpp"
#include "audio/sas_core.hpp"
#include "gpu/vulkan_renderer.hpp"
#include "install/installer.hpp"
#include "kernel/iso_image.hpp"
#include "mods/mhp3rd_mods.hpp"
#include "settings/settings.hpp"

#include "imgui.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mhp3rd::ui {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr float kFadeSeconds = 0.35f;
constexpr float kMusicFadeInSeconds = 2.0f;
constexpr float kMusicLevel = 0.75f;

// DATA.BIN entries of NPJB-40001 that hold the title's and the menus' 2D art.
constexpr std::uint32_t kMenuArt = 0x0FEE;  // letters, parchment and frame, the cart
constexpr std::uint32_t kInterface = 0x0FEA;  // the interface's parts: buttons, boxes, bars

// Colours the game gives its menu's sprites (ABGR as the GE has them).
constexpr ImU32 kShade = IM_COL32(0x75, 0x54, 0x2F, 0xFF);    // the diamonds, the highlight's shadow
constexpr ImU32 kInitial = IM_COL32(0x9E, 0x75, 0x5B, 0xFF);  // the initial of a row not chosen
constexpr ImU32 kInk = IM_COL32(0x43, 0x25, 0x13, 0xFF);      // the rows' text
constexpr ImU32 kChosen = IM_COL32(0xFF, 0x84, 0x00, 0xFF);   // the chosen answer of a question

// A sprite of the frame around the parchment: where it goes and the texels it
// shows, reversed for a mirrored one.
struct Piece {
    float x0, y0, x1, y1, u0, v0, u1, v1;
};
constexpr Piece kFramePieces[] = {
    {0, 0, 64, 48, 0, 0, 64, 48},
    {0, 48, 40, 64, 0, 48, 40, 64},
    {40, 48, 48, 56, 40, 48, 48, 56},
    {64, 0, 128, 16, 64, 40, 128, 56},
    {64, 16, 120, 24, 64, 56, 120, 64},
    {64, 24, 80, 40, 48, 48, 64, 64},
    {80, 24, 96, 32, 80, 16, 96, 24},
    {0, 64, 24, 80, 64, 0, 88, 16},
    {0, 80, 16, 88, 64, 16, 80, 24},
    {24, 64, 32, 72, 72, 16, 80, 24},
    {0, 88, 8, 104, 64, 24, 72, 40},
    {128, 0, 144, 16, 72, 24, 88, 40},
    {144, 0, 208, 8, 88, 24, 128, 32},
    {416, 0, 480, 48, 64, 0, 0, 48},
    {440, 48, 480, 64, 40, 48, 0, 64},
    {432, 48, 440, 56, 48, 48, 40, 56},
    {352, 0, 416, 16, 128, 40, 64, 56},
    {360, 16, 416, 24, 120, 56, 64, 64},
    {400, 24, 416, 40, 64, 48, 48, 64},
    {384, 24, 400, 32, 96, 16, 80, 24},
    {456, 64, 480, 80, 88, 0, 64, 16},
    {464, 80, 480, 88, 80, 16, 64, 24},
    {448, 64, 456, 72, 80, 16, 72, 24},
    {472, 88, 480, 104, 72, 24, 64, 40},
    {336, 0, 352, 16, 88, 24, 72, 40},
    {272, 0, 336, 8, 128, 24, 88, 32},
    {0, 224, 64, 272, 0, 48, 64, 0},
    {40, 216, 48, 224, 40, 56, 48, 48},
    {64, 256, 128, 272, 64, 56, 128, 40},
    {64, 248, 120, 256, 64, 64, 120, 56},
    {64, 232, 80, 248, 48, 64, 64, 48},
    {80, 240, 96, 248, 80, 24, 96, 16},
    {0, 192, 24, 208, 64, 16, 88, 0},
    {0, 184, 16, 192, 64, 24, 80, 16},
    {24, 200, 32, 208, 72, 24, 80, 16},
    {0, 168, 8, 184, 64, 40, 72, 24},
    {128, 256, 144, 272, 72, 40, 88, 24},
    {144, 264, 208, 272, 88, 32, 128, 24},
    {416, 224, 480, 272, 64, 48, 0, 0},
    {440, 208, 480, 224, 40, 64, 0, 48},
    {432, 216, 440, 224, 48, 56, 40, 48},
    {352, 256, 416, 272, 128, 56, 64, 40},
    {360, 248, 416, 256, 120, 64, 64, 56},
    {400, 232, 416, 248, 64, 64, 48, 48},
    {384, 240, 400, 248, 96, 24, 80, 16},
    {456, 192, 480, 208, 88, 16, 64, 0},
    {464, 184, 480, 192, 80, 24, 64, 16},
    {448, 200, 456, 208, 80, 24, 72, 16},
    {472, 168, 480, 184, 72, 40, 64, 24},
    {336, 256, 352, 272, 88, 40, 72, 24},
    {272, 264, 336, 272, 128, 32, 88, 24},
};

// The scene beside the menu, as the game animates it at 30 frames a second
// (traced over a few seconds of its GAME MENU): the aptonoth pulling the cart
// walks on the spot in four poses of 8 frames each, on a hill that stays put,
// while the grass on the hill, the rocks and a cloud behind it pass by along
// arcs, and a dragonfly comes and goes.
constexpr Piece kHill[] = {
    {300, 180, 348, 204, 208, 0, 256, 24},
    {276, 180, 300, 204, 232, 24, 256, 48},
    {260, 188, 276, 204, 216, 24, 232, 40},
    {252, 196, 260, 204, 224, 40, 232, 48},
    {316, 172, 348, 180, 216, 48, 248, 56},
    {348, 180, 396, 204, 256, 0, 208, 24},
    {392, 180, 416, 204, 256, 24, 232, 48},
    {416, 188, 432, 204, 232, 24, 216, 40},
    {432, 196, 440, 204, 232, 40, 224, 48},
    {348, 172, 380, 180, 248, 48, 216, 56}
};
constexpr Piece kWalk0[] = {
        {276, 136, 324, 168, 0, 0, 48, 32},
        {324, 144, 340, 184, 48, 8, 64, 48},
        {292, 168, 324, 184, 16, 32, 48, 48},
        {340, 104, 412, 176, 64, 16, 136, 88},
        {356, 88, 396, 104, 80, 0, 120, 16},
        {412, 120, 420, 136, 136, 24, 144, 40},
        {364, 176, 380, 184, 0, 32, 16, 40},
        {380, 176, 396, 184, 0, 40, 16, 48}
};
constexpr Piece kWalk1[] = {
        {276, 136, 340, 168, 0, 48, 64, 80},
        {292, 168, 340, 184, 16, 80, 64, 96},
        {348, 104, 404, 176, 144, 16, 200, 88},
        {340, 128, 348, 176, 136, 40, 144, 88},
        {356, 88, 412, 104, 152, 0, 208, 16},
        {404, 104, 412, 152, 200, 16, 208, 64},
        {412, 112, 420, 144, 208, 24, 216, 56},
        {420, 128, 428, 136, 216, 40, 224, 48},
        {364, 176, 380, 184, 0, 80, 16, 88},
        {380, 176, 396, 184, 0, 88, 16, 96},
        {364, 80, 372, 88, 136, 0, 144, 8}
};
constexpr Piece kWalk2[] = {
        {276, 136, 292, 168, 0, 96, 16, 128},
        {292, 136, 324, 184, 16, 96, 48, 144},
        {324, 144, 340, 184, 48, 104, 64, 144},
        {340, 152, 404, 176, 64, 160, 128, 184},
        {364, 176, 396, 184, 48, 96, 80, 104},
        {348, 112, 416, 152, 72, 120, 140, 160},
        {340, 128, 350, 160, 64, 136, 74, 168},
        {348, 104, 356, 112, 72, 112, 80, 120},
        {356, 80, 388, 112, 80, 88, 112, 120},
        {388, 96, 412, 112, 112, 104, 136, 120},
        {416, 144, 420, 152, 140, 152, 144, 160}
};
constexpr Piece kWalk3[] = {
        {276, 136, 336, 168, 0, 144, 60, 176},
        {292, 168, 340, 176, 16, 176, 64, 184},
        {300, 176, 324, 184, 136, 88, 160, 96},
        {340, 128, 404, 176, 144, 120, 208, 168},
        {348, 104, 412, 128, 152, 96, 216, 120},
        {404, 128, 412, 152, 208, 120, 216, 144},
        {356, 96, 412, 104, 160, 88, 216, 96},
        {356, 80, 372, 88, 120, 88, 136, 96},
        {364, 88, 388, 96, 128, 96, 152, 104},
        {412, 120, 420, 136, 112, 88, 120, 104},
        {420, 128, 428, 136, 120, 96, 128, 104},
        {380, 176, 396, 184, 136, 112, 152, 120},
        {364, 176, 380, 184, 136, 104, 152, 112},
        {336, 160, 340, 168, 60, 168, 64, 176},
        {336, 144, 340, 152, 140, 136, 144, 144}
};
constexpr std::span<const Piece> kWalk[] = {kWalk0, kWalk1, kWalk2, kWalk3};
constexpr float kWalkFrames = 8.0f;

// Something that passes along an arc about (cx, cy): at `radius`, from angle
// `from` to `to` (degrees, clockwise from straight up) at `speed` degrees a
// frame, turned with the arc, fading in and out over `fade` degrees at the
// ends and gone for the rest of its `cycle`.
struct Passer {
    float u0, v0, u1, v1;  // its texels
    float cx, cy, radius;
    float phase;           // its angle at frame 0
    float speed, from, to, cycle, fade;
    std::uint32_t alpha;
};
constexpr Passer kPassers[] = {
    // Rocks behind the hill, translucent.
    {176, 200, 208, 256, 339, 338, 244, -13.0f, 0.0404f, -16.5f, 16.8f, 33.3f, 2.0f, 128},
    {208, 176, 256, 256, 339, 338, 255, -4.0f, 0.0404f, -16.5f, 16.8f, 33.3f, 2.0f, 128},
    {216, 56, 248, 136, 339, 338, 250, 4.5f, 0.0404f, -16.5f, 16.8f, 33.3f, 2.0f, 128},
    {0, 208, 24, 256, 339, 338, 232, 12.5f, 0.0404f, -16.5f, 16.8f, 33.3f, 2.0f, 128},
};
constexpr Passer kCloud{0, 184, 80, 208, 343, 366, 286, -10.0f, 0.078f, -21.0f, 21.0f, 46.0f, 3.0f, 180};
// The grass on the hill: the two drawn behind the cart, then the two in front.
constexpr Passer kGrassBehind[] = {
    {80, 184, 112, 208, 348, 378, 210, 23.4f, 0.25f, -24.0f, 25.0f, 54.0f, 2.0f, 255},
    {64, 0, 80, 16, 348, 378, 206, -3.9f, 0.25f, -24.0f, 25.0f, 54.0f, 2.0f, 255},
};
constexpr Passer kGrassFront[] = {
    {0, 128, 16, 144, 348, 378, 195, -8.3f, 0.25f, -24.0f, 25.0f, 54.0f, 2.0f, 255},
    {56, 0, 64, 8, 348, 378, 198, 2.2f, 0.25f, -24.0f, 25.0f, 54.0f, 2.0f, 255},
};

std::uint16_t le16(std::span<const std::uint8_t> b, std::size_t at) {
    return static_cast<std::uint16_t>(b[at] | (b[at + 1u] << 8u));
}
std::uint32_t le32(std::span<const std::uint8_t> b, std::size_t at) {
    return static_cast<std::uint32_t>(b[at] | (b[at + 1u] << 8u) | (b[at + 2u] << 16u) | (b[at + 3u] << 24u));
}

float ease(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

ImU32 with_alpha(ImU32 color, float alpha) {
    const auto a = static_cast<float>((color >> IM_COL32_A_SHIFT) & 0xFFu) * std::clamp(alpha, 0.0f, 1.0f);
    return (color & ~IM_COL32_A_MASK) | (static_cast<ImU32>(a) << IM_COL32_A_SHIFT);
}

std::vector<std::uint8_t> read_disc_file(IsoImage &iso, const char *path, std::uint32_t limit) {
    const auto entry = iso.find(path);
    if (!entry || entry->directory || entry->size == 0u || entry->size > limit) return {};
    std::vector<std::uint8_t> bytes(entry->size);
    bytes.resize(iso.read(static_cast<std::uint64_t>(entry->lba) * IsoImage::kSectorSize, bytes));
    return bytes;
}

// One of the game's menu sounds, as its SAS voices play it: VAG samples in
// one of its sound banks in DATA.BIN, each at a pitch (0x1000 is 44100 Hz) and
// a volume per side (0x1000 is full), the decide sound as a pair of voices
// panned apart. Offsets and settings as traced with MHP3RD_TRACE_SAS.
struct Voice {
    std::uint32_t entry, offset, size;
    std::int32_t left, right;
};
struct SoundEffect {
    Voice voices[2];
    std::size_t count;
    std::uint32_t pitch;
};
constexpr SoundEffect kCursorSound{{{0x10CA, 0xA4A0, 4304, 1400, 1400}}, 1, 0x8B5};
constexpr SoundEffect kDecideSound{{{0x10CA, 0x0000, 21072, 1980, 0}, {0x10CA, 0x5250, 21072, 24, 1979}}, 2, 0x8B5};
constexpr SoundEffect kCancelSound{{{0x10C4, 0x1590, 2240, 1262, 1262}}, 1, 0x5CE};

// The launcher's sound: the disc's menu music on a loop and the game's own
// menu sounds, mixed and fed to the audio device a frame at a time.
class Mixer {
public:
    // Decodes SND0.AT3 whole: a short ATRAC3 loop of a few hundred kilobytes.
    bool load_music(std::span<const std::uint8_t> file);
    // Decodes a sound from its bank, resampled to the device's 44100 Hz
    // stereo. Empty when the bank is not what this release has.
    static std::vector<std::int16_t> load_sound(const SoundEffect &effect);
    void play(const std::vector<std::int16_t> &sound);
    // Whether a sound is still playing.
    [[nodiscard]] bool busy() const noexcept { return !playing_.empty(); }
    // Mixes what the device needs since the last call, the music at `gain`.
    void pump(float gain);

private:
    std::vector<std::int16_t> music_;  // interleaved stereo at 44100 Hz
    std::size_t position_{};           // the next frame of music_ to play
    struct Playing {
        const std::vector<std::int16_t> *sound;
        std::size_t frame;
    };
    std::vector<Playing> playing_;
    std::uint64_t cursor_{};           // the audio sink's frame index
    std::optional<Clock::time_point> last_;
    double owed_{};                    // frames of real time not mixed yet
    std::vector<std::int32_t> sum_;
    std::vector<std::int16_t> out_;
};

bool Mixer::load_music(std::span<const std::uint8_t> file) {
    if (!audio::AtracDecoder::available() || file.size() < 12u) return false;
    if (le32(file, 0u) != 0x46464952u || le32(file, 8u) != 0x45564157u) return false;  // "RIFF", "WAVE"
    unsigned channels = 0;
    std::uint32_t rate = 0;
    unsigned block_align = 0;
    std::optional<audio::AtracCodec> codec;
    std::vector<std::uint8_t> extradata;
    std::size_t data = 0;
    std::size_t data_size = 0;
    std::uint32_t skip = 0;
    for (std::size_t at = 12u; at + 8u <= file.size();) {
        const std::uint32_t id = le32(file, at);
        const std::uint32_t size = le32(file, at + 4u);
        const std::size_t body = at + 8u;
        if (id == 0x61746164u) {  // "data"
            data = body;
            data_size = std::min<std::size_t>(size, file.size() - body);
            break;
        }
        if (body + size > file.size()) return false;
        if (id == 0x20746D66u && size >= 16u) {  // "fmt "
            const std::uint16_t tag = le16(file, body);
            channels = le16(file, body + 2u);
            rate = le32(file, body + 4u);
            block_align = le16(file, body + 12u);
            if (tag == 0x0270u) {
                codec = audio::AtracCodec::Atrac3;
                if (size >= 32u) extradata.assign(file.begin() + static_cast<std::ptrdiff_t>(body + 18u),
                                                  file.begin() + static_cast<std::ptrdiff_t>(body + 32u));
            } else if (tag == 0xFFFEu) {
                codec = audio::AtracCodec::Atrac3Plus;
            }
        } else if (id == 0x74636166u && size >= 8u) {  // "fact": samples, then the offset of the first
            skip = le32(file, body + 4u);
        }
        at = body + size + (size & 1u);
    }
    if (!codec || data == 0u || channels == 0u || channels > 2u || block_align == 0u || rate != audio::kSampleRate)
        return false;

    audio::AtracDecoder decoder;
    if (!decoder.open(*codec, channels, block_align, extradata)) return false;
    std::vector<std::int16_t> frame(audio::atrac_frame_samples(*codec) * 2u);
    for (std::size_t at = data; at + block_align <= data + data_size; at += block_align) {
        const std::size_t samples = decoder.decode(file.subspan(at, block_align), frame.data());
        if (samples == 0u) return false;
        music_.insert(music_.end(), frame.begin(), frame.begin() + static_cast<std::ptrdiff_t>(samples * 2u));
    }
    // The decoder's delay and the encoder's own padding come first.
    const std::size_t lead = std::min<std::size_t>(music_.size() / 2u, skip + 69u);
    music_.erase(music_.begin(), music_.begin() + static_cast<std::ptrdiff_t>(lead * 2u));
    if (music_.size() < 2u * audio::kSampleRate / 4u) music_.clear();
    return !music_.empty();
}

std::vector<std::int16_t> Mixer::load_sound(const SoundEffect &effect) {
    const double rate = 44100.0 * effect.pitch / 4096.0;
    std::vector<double> left;
    std::vector<double> right;
    for (std::size_t v = 0; v < effect.count; ++v) {
        const Voice &voice = effect.voices[v];
        const std::vector<std::uint8_t> bank = mods::entry(voice.entry);
        if (voice.offset + voice.size > bank.size()) return {};
        const std::vector<std::int16_t> mono =
            audio::decode_vag(std::span<const std::uint8_t>(bank).subspan(voice.offset, voice.size));
        if (mono.empty()) return {};
        // To 44100 Hz by linear interpolation, as the SAS mixer steps its
        // voices by their pitch.
        const auto frames = static_cast<std::size_t>(static_cast<double>(mono.size()) * 44100.0 / rate);
        if (left.size() < frames) {
            left.resize(frames, 0.0);
            right.resize(frames, 0.0);
        }
        for (std::size_t i = 0; i < frames; ++i) {
            const double at = static_cast<double>(i) * rate / 44100.0;
            const auto index = static_cast<std::size_t>(at);
            const double t = at - static_cast<double>(index);
            const double a = mono[std::min(index, mono.size() - 1u)];
            const double b = mono[std::min(index + 1u, mono.size() - 1u)];
            const double sample = a + (b - a) * t;
            left[i] += sample * voice.left / 4096.0;
            right[i] += sample * voice.right / 4096.0;
        }
    }
    std::vector<std::int16_t> stereo(left.size() * 2u);
    for (std::size_t i = 0; i < left.size(); ++i) {
        stereo[i * 2u] = static_cast<std::int16_t>(std::clamp(left[i], -32768.0, 32767.0));
        stereo[i * 2u + 1u] = static_cast<std::int16_t>(std::clamp(right[i], -32768.0, 32767.0));
    }
    return stereo;
}

void Mixer::play(const std::vector<std::int16_t> &sound) {
    if (!sound.empty()) playing_.push_back({&sound, 0u});
}

void Mixer::pump(float gain) {
    const Clock::time_point now = Clock::now();
    // The first call primes the device with a moment of sound; after a stall
    // (a window drag) the sink restarts the stream rather than catch up.
    const double elapsed = last_ ? std::min(std::chrono::duration<double>(now - *last_).count(), 0.1) : 0.05;
    last_ = now;
    owed_ += elapsed * audio::kSampleRate;
    const auto count = static_cast<std::size_t>(owed_);
    owed_ -= static_cast<double>(count);
    if (count == 0u || (music_.empty() && playing_.empty())) return;
    sum_.assign(count * 2u, 0);
    const auto music_gain = static_cast<std::int32_t>(std::clamp(gain, 0.0f, 1.0f) * 4096.0f);
    if (!music_.empty()) {
        const std::size_t frames = music_.size() / 2u;
        for (std::size_t i = 0; i < count; ++i) {
            sum_[i * 2u] += music_[position_ * 2u] * music_gain >> 12;
            sum_[i * 2u + 1u] += music_[position_ * 2u + 1u] * music_gain >> 12;
            position_ = (position_ + 1u) % frames;
        }
    }
    for (Playing &p : playing_) {
        const std::size_t frames = p.sound->size() / 2u;
        for (std::size_t i = 0; i < count && p.frame < frames; ++i, ++p.frame) {
            sum_[i * 2u] += (*p.sound)[p.frame * 2u];
            sum_[i * 2u + 1u] += (*p.sound)[p.frame * 2u + 1u];
        }
    }
    std::erase_if(playing_, [](const Playing &p) { return p.frame >= p.sound->size() / 2u; });
    out_.resize(sum_.size());
    for (std::size_t i = 0; i < sum_.size(); ++i) out_[i] = static_cast<std::int16_t>(std::clamp(sum_[i], -32768, 32767));
    audio::AudioSink::instance().mix(cursor_, out_.data(), count, 0x8000u, 0x8000u);
}

// The game's 480x272 screen in the window.
struct Screen {
    ImVec2 window;
    float scale{};
    ImVec2 origin;  // the top left of the middle 480x272

    static Screen of(ImVec2 window) {
        Screen s;
        s.window = window;
        s.scale = std::min(window.x / 480.0f, window.y / 272.0f);
        s.origin = {std::round((window.x - 480.0f * s.scale) * 0.5f), std::round((window.y - 272.0f * s.scale) * 0.5f)};
        return s;
    }
    [[nodiscard]] ImVec2 at(float x, float y) const { return {origin.x + x * scale, origin.y + y * scale}; }
    // A point of the frame, kept to the nearest corner of the window.
    [[nodiscard]] ImVec2 cornered(float x, float y, bool right, bool bottom) const {
        return {right ? window.x - (480.0f - x) * scale : x * scale,
                bottom ? window.y - (272.0f - y) * scale : y * scale};
    }
};

// Texel coordinates of a letter of the menu's alphabet: A to J, K to T and
// U to Z in three rows of 24-pixel cells.
ImVec2 letter_cell(char c) {
    const int index = std::clamp(c - 'A', 0, 25);
    return {static_cast<float>(index % 10 * 24), static_cast<float>(index / 10 * 24)};
}

enum class Item { Play, Settings, Textures, Mods, Saves, Count };
constexpr const char *kLabels[] = {"PLAY", "SETTINGS", "TEXTURES", "MODS", "SAVE DATA"};
// The rows as the game spaces its own: a row every 33 pixels from 58 (the
// game's four rows are 36 apart; five need a little less).
constexpr float kFirstRow = 58.0f;
constexpr float kRowPitch = 33.0f;

class Launcher {
public:
    explicit Launcher(const fs::path &disc_image);
    // One frame; false once the launcher is done.
    bool frame();
    [[nodiscard]] LauncherChoice choice() const noexcept { return choice_; }

private:
    [[nodiscard]] bool art_ready() const { return menu_ && frame_ && interface_ && footer_ && box_; }
    void draw_background(ImDrawList *draw, const Screen &s) const;
    void draw_title(ImDrawList *draw, const Screen &s, std::string_view title) const;
    void draw_cart(ImDrawList *draw, const Screen &s) const;
    void draw_dragonfly(ImDrawList *draw, const Screen &s, float frames) const;
    void draw_rows(ImDrawList *draw, const Screen &s) const;
    void draw_footer(ImDrawList *draw, const Screen &s) const;
    void draw_question(ImDrawList *draw, const Screen &s, std::string_view question) const;
    void draw_plain(ImDrawList *draw, const Screen &s) const;
    void icon(ImDrawList *draw, const Screen &s, float x, float y, float u, float v, float size) const;
    void handle_input(const Screen &s);
    void activate(Item item);
    void leave(LauncherChoice choice);

    art::Sheet menu_;       // the alphabet, the diamonds, the highlight
    art::Sheet frame_;      // parchment and the frame's pieces
    art::Sheet cart_;       // the cart's picture
    art::Sheet cart_rock_;  // ...its cloud, which the game draws with the picture's first palette
    art::Sheet interface_;  // button glyphs and arrows (palette 10), from (128, 0)
    art::Sheet arrows_;     // the question's arrows (palette 9), from (160, 16)
    art::Sheet footer_;     // the bar at the bottom (palette 7)
    art::Sheet box_;        // the question's box (palette 6), from (0, 88)
    art::Text text_;
    Mixer mixer_;
    bool music_{};
    std::vector<std::int16_t> cursor_sound_;
    std::vector<std::int16_t> decide_sound_;
    std::vector<std::int16_t> cancel_sound_;

    int row_{};
    bool asking_{};         // "Do you want to quit?" is up
    bool answer_yes_{};
    std::optional<double> started_;
    std::optional<double> leaving_since_;
    LauncherChoice choice_{LauncherChoice::Play};
    ImVec2 last_mouse_{-1.0f, -1.0f};
};

Launcher::Launcher(const fs::path &disc_image) {
    // The game's sprites are sampled texel for texel; enlarging them by the
    // window's scale keeps them as sharp.
    const float window = std::max(ImGui::GetIO().DisplaySize.y / 272.0f, 1.0f);
    const int scale = std::clamp(static_cast<int>(std::ceil(window)), 2, 8);
    menu_ = art::load_sheet(kMenuArt, 0u, 0u, {}, scale);
    frame_ = art::load_sheet(kMenuArt, 1u, 0u, {}, scale);
    cart_ = art::load_sheet(kMenuArt, 2u, 3u, {}, scale);
    cart_rock_ = art::load_sheet(kMenuArt, 2u, 0u, {0, 184, 80, 24}, scale);
    interface_ = art::load_sheet(kInterface, 1u, 10u, {128, 0, 64, 32}, scale);
    arrows_ = art::load_sheet(kInterface, 1u, 9u, {160, 16, 32, 32}, scale);
    footer_ = art::load_sheet(kInterface, 1u, 7u, {32, 104, 32, 16}, scale);
    box_ = art::load_sheet(kInterface, 1u, 6u, {0, 88, 64, 36}, scale);
    (void)text_.ready();
    if (!disc_image.empty() && settings::current().launcher_music && audio::AudioSink::instance().has_device()) {
        try {
            IsoImage iso(disc_image);
            music_ = mixer_.load_music(read_disc_file(iso, "PSP_GAME/SND0.AT3", 8u << 20u));
        } catch (const std::exception &e) {
            std::cout << "[launcher] cannot read the disc's music: " << e.what() << "\n";
        }
    }
    if (audio::AudioSink::instance().has_device()) {
        cursor_sound_ = Mixer::load_sound(kCursorSound);
        decide_sound_ = Mixer::load_sound(kDecideSound);
        cancel_sound_ = Mixer::load_sound(kCancelSound);
    }
    std::cout << "[launcher] the game's art " << (art_ready() ? "from DATA.BIN" : "not found; a plain screen instead")
              << ", music " << (music_ ? "from the disc" : "off") << ", sounds "
              << (cursor_sound_.empty() ? "off" : "from DATA.BIN") << std::endl;
}

void Launcher::leave(LauncherChoice choice) {
    if (leaving_since_) return;
    choice_ = choice;
    leaving_since_ = ImGui::GetTime();
    std::cout << (choice == LauncherChoice::Play ? "[launcher] play" : "[launcher] quit") << std::endl;
}

void Launcher::activate(Item item) {
    switch (item) {
    case Item::Play: leave(LauncherChoice::Play); break;
    case Item::Settings: open_launcher_menu(MenuTab::Video); break;
    case Item::Textures: open_launcher_menu(MenuTab::Video, MenuFocus::TexturePack); break;
    case Item::Mods: open_launcher_menu(MenuTab::Mods); break;
    case Item::Saves: open_launcher_menu(MenuTab::System, MenuFocus::Saves); break;
    case Item::Count: break;
    }
}

void Launcher::draw_background(ImDrawList *draw, const Screen &s) const {
    // Parchment, 128x64 at a time, over the whole window, lined up with the
    // middle screen as the game lines it up with its own.
    const float w = 128.0f * s.scale;
    const float h = 64.0f * s.scale;
    const float x0 = s.origin.x - std::ceil(s.origin.x / w) * w;
    const float y0 = s.origin.y - std::ceil(s.origin.y / h) * h;
    for (float y = y0; y < s.window.y; y += h)
        for (float x = x0; x < s.window.x; x += w)
            draw->AddImage(frame_.ref(), {x, y}, {x + w, y + h}, frame_.uv(0, 64), frame_.uv(128, 128));
    for (const Piece &p : kFramePieces) {
        const bool right = p.x0 + p.x1 > 480.0f;
        const bool bottom = p.y0 + p.y1 > 272.0f;
        draw->AddImage(frame_.ref(), s.cornered(p.x0, p.y0, right, bottom), s.cornered(p.x1, p.y1, right, bottom),
                       frame_.uv(p.u0, p.v0), frame_.uv(p.u1, p.v1));
    }
}

// A title in the menu's lettering, centred at the top: each letter on a
// diamond with its caps, 37 pixels apart, a space 17 more.
void Launcher::draw_title(ImDrawList *draw, const Screen &s, std::string_view title) const {
    float width = 0.0f;
    for (const char c : title) width += c == ' ' ? 17.0f : 37.0f;
    float x = 240.0f - (width - 37.0f + 40.0f) * 0.5f;
    const auto piece = [&](float px, float py, float w, float h, float u, float v) {
        draw->AddImage(menu_.ref(), s.at(px, py), s.at(px + w, py + h), menu_.uv(u, v), menu_.uv(u + w, v + h));
    };
    for (const char c : title) {
        if (c == ' ') {
            x += 17.0f;
            continue;
        }
        piece(x + 8.0f, 12.0f, 24.0f, 24.0f, 144.0f, 48.0f);   // diamond
        piece(x + 32.0f, 12.0f, 8.0f, 24.0f, 248.0f, 0.0f);    // its right point
        piece(x, 12.0f, 8.0f, 24.0f, 240.0f, 0.0f);            // its left point
        piece(x + 13.0f, 36.0f, 16.0f, 8.0f, 240.0f, 32.0f);   // below
        piece(x + 13.0f, 4.0f, 16.0f, 8.0f, 240.0f, 24.0f);    // above
        const ImVec2 cell = letter_cell(c);
        piece(x + 8.0f, 12.0f, 24.0f, 24.0f, cell.x, cell.y);
        x += 37.0f;
    }
}

// `sheet`'s texels (u0, v0)-(u1, v1) as a quad of their size centred on
// (x, y), turned by `degrees`.
void turned(ImDrawList *draw, const Screen &s, const art::Sheet &sheet, float u0, float v0, float u1, float v1,
            float x, float y, float degrees, ImU32 color) {
    const float angle = degrees * 3.14159265f / 180.0f;
    const float c = std::cos(angle);
    const float n = std::sin(angle);
    const float hw = (u1 - u0) * 0.5f;
    const float hh = (v1 - v0) * 0.5f;
    const auto corner = [&](float dx, float dy) { return s.at(x + dx * c - dy * n, y + dx * n + dy * c); };
    draw->AddImageQuad(sheet.ref(), corner(-hw, -hh), corner(hw, -hh), corner(hw, hh), corner(-hw, hh),
                       sheet.uv(u0, v0), sheet.uv(u1, v0), sheet.uv(u1, v1), sheet.uv(u0, v1), color);
}

void pass(ImDrawList *draw, const Screen &s, const art::Sheet &sheet, const Passer &p, float frames, float dv = 0.0f) {
    const float span = p.to - p.from;
    // Its angle, wrapped into its cycle, which starts at .
    float angle = std::fmod(p.phase - p.from + p.speed * frames, p.cycle);
    if (angle < 0.0f) angle += p.cycle;
    if (angle > span) return;
    const float fade = std::min({1.0f, angle / p.fade, (span - angle) / p.fade});
    angle += p.from;
    const float r = angle * 3.14159265f / 180.0f;
    const float x = p.cx + p.radius * std::sin(r);
    const float y = p.cy - p.radius * std::cos(r);
    turned(draw, s, sheet, p.u0, p.v0 - dv, p.u1, p.v1 - dv, x, y, angle,
           IM_COL32(255, 255, 255, static_cast<int>(static_cast<float>(p.alpha) * fade)));
}

void Launcher::draw_cart(ImDrawList *draw, const Screen &s) const {
    // The game's frames since the launcher came up.
    const auto frames = static_cast<float>((ImGui::GetTime() - started_.value_or(0.0)) * 30.0);
    const auto piece = [&](const Piece &p) {
        draw->AddImage(cart_.ref(), s.at(p.x0, p.y0), s.at(p.x1, p.y1), cart_.uv(p.u0, p.v0), cart_.uv(p.u1, p.v1));
    };
    for (const Passer &p : kPassers) pass(draw, s, cart_, p, frames);
    pass(draw, s, cart_rock_, kCloud, frames, 184.0f);
    for (const Piece &p : kHill) piece(p);
    for (const Passer &p : kGrassBehind) pass(draw, s, cart_, p, frames);
    const auto pose = static_cast<std::size_t>(frames / kWalkFrames) % std::size(kWalk);
    for (const Piece &p : kWalk[pose]) piece(p);
    for (const Passer &p : kGrassFront) pass(draw, s, cart_, p, frames);
    draw_dragonfly(draw, s, frames);
}

// The dragonfly: in from the lower left, a while hovering over the cart, then
// off to the right, and back after a pause; 250 frames a round.
void Launcher::draw_dragonfly(ImDrawList *draw, const Screen &s, float frames) const {
    const float t = std::fmod(frames, 250.0f);
    struct Key {
        float t, x, y, alpha;
    };
    constexpr Key keys[] = {{0, 262, 112, 0},   {10, 274, 106, 1},  {40, 340, 71, 1},   {50, 352, 72, 1},
                            {180, 355, 72, 1},  {200, 389, 78, 1},  {212, 433, 85, 0},  {250, 433, 85, 0}};
    std::size_t i = 0;
    while (i + 2u < std::size(keys) && t >= keys[i + 1u].t) ++i;
    const Key &a = keys[i];
    const Key &b = keys[i + 1u];
    const float k = std::clamp((t - a.t) / (b.t - a.t), 0.0f, 1.0f);
    float x = a.x + (b.x - a.x) * k;
    float y = a.y + (b.y - a.y) * k;
    const float alpha = a.alpha + (b.alpha - a.alpha) * k;
    if (alpha <= 0.01f) return;
    // While it hovers it darts about a little, as the game's does.
    if (t > 50.0f && t < 180.0f) {
        x += 3.0f * std::sin(t * 0.21f) + 2.0f * std::sin(t * 0.53f);
        y += 5.0f * std::sin(t * 0.17f + 1.0f) + 3.0f * std::sin(t * 0.61f);
    }
    turned(draw, s, cart_, 121, 1, 135, 14, x, y, 55.0f + 5.0f * std::sin(t * 0.3f),
           IM_COL32(255, 255, 255, static_cast<int>(255.0f * alpha)));
}

void Launcher::draw_rows(ImDrawList *draw, const Screen &s) const {
    const auto piece = [&](float x0, float y0, float x1, float y1, float u0, float v0, float u1, float v1, ImU32 c) {
        draw->AddImage(menu_.ref(), s.at(x0, y0), s.at(x1, y1), menu_.uv(u0, v0), menu_.uv(u1, v1), c);
    };
    // The window's left edge, in the game's pixels: the highlight reaches it.
    const float left = -s.origin.x / s.scale;
    for (int i = 0; i < static_cast<int>(Item::Count); ++i) {
        const float y = kFirstRow + kRowPitch * static_cast<float>(i);
        const std::string_view label = kLabels[i];
        const ImVec2 cell = letter_cell(label.front());
        if (i == row_) {
            // The highlight: a shadow a pixel lower, then the band, which ends
            // in a point.
            for (int layer = 0; layer < 2; ++layer) {
                const float top = y - (layer == 0 ? 6.0f : 7.0f);
                const ImU32 c = layer == 0 ? kShade : IM_COL32_WHITE;
                for (float x = 162.0f; x + 32.0f > left; x -= 32.0f)
                    piece(x, top, x + 32.0f, top + 30.0f, 0.0f, 73.0f, 32.0f, 103.0f, c);
                piece(194.0f, top, layer == 0 ? 239.0f : 238.0f, top + 30.0f, layer == 0 ? 3.0f : 4.0f, 73.0f, 48.0f,
                      103.0f, c);
            }
            piece(44.0f, y - 6.0f, 72.0f, y + 22.0f, cell.x, cell.y, cell.x + 24.0f, cell.y + 24.0f, IM_COL32_WHITE);
        } else {
            piece(40.0f, y - 9.0f, 56.0f, y + 23.0f, 48.0f, 72.0f, 32.0f, 104.0f, kShade);
            piece(56.0f, y - 9.0f, 72.0f, y + 23.0f, 32.0f, 72.0f, 48.0f, 104.0f, kShade);
            piece(45.0f, y - 4.0f, 67.0f, y + 18.0f, cell.x, cell.y, cell.x + 24.0f, cell.y + 24.0f, kInitial);
        }
        text_.draw(draw, s.at(74.0f, y), s.scale, art::kMenuText, label.substr(1), kInk);
    }
}

void Launcher::icon(ImDrawList *draw, const Screen &s, float x, float y, float u, float v, float size) const {
    // Glyphs from (128, 0) of the interface's parts: circle, cross, triangle,
    // square, and the D-pad below the cross.
    const ImVec2 a = s.cornered(x, y, true, true);
    const ImVec2 b = s.cornered(x + size, y + size, true, true);
    draw->AddImage(interface_.ref(), a, b, interface_.uv(u - 128.0f, v), interface_.uv(u - 128.0f + size, v + size));
}

// The bar at the bottom, its pattern repeated across the window, and the
// buttons that work here at its right, as the game has its hints.
void Launcher::draw_footer(ImDrawList *draw, const Screen &s) const {
    const float w = 32.0f * s.scale;
    const float top = s.window.y - 16.0f * s.scale;
    for (float x = 0.0f; x < s.window.x; x += w)
        draw->AddImage(footer_.ref(), {x, top}, {x + w, s.window.y}, footer_.uv(0, 0), footer_.uv(32, 16));
    // From the right: Back, Enter, Select, each a glyph and a word.
    struct Hint {
        float u, v, size;
        const char *word;
    };
    const Hint shown[] = {{144.0f, 16.0f, 15.0f, "Select"}, {129.0f, 1.0f, 14.0f, "Enter"}, {145.0f, 1.0f, 14.0f, "Back"}};
    float x = 472.0f;
    for (int i = 2; i >= 0; --i) {
        const Hint &h = shown[i];
        const float word = art::Text::width(art::kSmallText, h.word);
        x -= word;
        const ImVec2 at = s.cornered(x, 258.0f, true, true);
        text_.draw(draw, at, s.scale, art::kSmallText, h.word, IM_COL32_WHITE);
        x -= 3.0f + h.size;
        icon(draw, s, x, h.size == 15.0f ? 257.0f : 257.0f, h.u, h.v, h.size);
        x -= 9.0f;
    }
}

// A question in the game's box at the bottom, with Yes and No under it.
void Launcher::draw_question(ImDrawList *draw, const Screen &s, std::string_view question) const {
    const auto part = [&](float x0, float y0, float x1, float y1, float u0, float v0, float u1, float v1) {
        draw->AddImage(box_.ref(), s.at(x0, y0), s.at(x1, y1), box_.uv(u0, v0 - 88.0f), box_.uv(u1, v1 - 88.0f));
    };
    for (float x = 16.0f; x < 464.0f; x += 32.0f)
        for (float y = 215.0f; y < 253.0f; y += 8.0f) {
            const float x1 = std::min(x + 32.0f, 464.0f);
            const float y1 = std::min(y + 8.0f, 253.0f);
            part(x, y, x1, y1, 0.0f, 104.0f, x1 - x, 104.0f + (y1 - y));
        }
    part(14.0f, 213.0f, 465.0f, 222.0f, 24.0f, 88.0f, 25.0f, 97.0f);    // top edge
    part(14.0f, 245.0f, 465.0f, 254.0f, 24.0f, 97.0f, 25.0f, 88.0f);    // bottom edge
    part(14.0f, 214.0f, 46.0f, 254.0f, 32.0f, 122.0f, 64.0f, 123.0f);   // left side
    part(434.0f, 214.0f, 466.0f, 254.0f, 64.0f, 122.0f, 32.0f, 123.0f); // right side
    part(14.0f, 213.0f, 46.0f, 222.0f, 0.0f, 113.0f, 32.0f, 122.0f);    // corners
    part(434.0f, 213.0f, 466.0f, 222.0f, 32.0f, 113.0f, 0.0f, 122.0f);
    part(14.0f, 245.0f, 46.0f, 254.0f, 0.0f, 122.0f, 32.0f, 113.0f);
    part(434.0f, 245.0f, 466.0f, 254.0f, 32.0f, 122.0f, 0.0f, 113.0f);

    const float qx = 240.0f - art::Text::width(art::kSmallText, question) * 0.5f;
    text_.draw(draw, s.at(std::floor(qx), 220.0f), s.scale, art::kSmallText, question, IM_COL32_WHITE);
    // Yes and No, the chosen one orange between two arrows.
    const float yes = 202.0f;
    const float no = 265.0f;
    text_.draw(draw, s.at(yes, 236.0f), s.scale, art::kSmallText, "Yes", answer_yes_ ? kChosen : IM_COL32_WHITE);
    text_.draw(draw, s.at(no, 236.0f), s.scale, art::kSmallText, "No", answer_yes_ ? IM_COL32_WHITE : kChosen);
    const float left = answer_yes_ ? yes - 14.0f : no - 14.0f;
    const float right = answer_yes_ ? yes + art::Text::width(art::kSmallText, "Yes") + 1.0f
                                    : no + art::Text::width(art::kSmallText, "No") + 1.0f;
    draw->AddImage(arrows_.ref(), s.at(left, 236.0f), s.at(left + 13.0f, 249.0f), arrows_.uv(0, 0), arrows_.uv(13, 13));
    draw->AddImage(arrows_.ref(), s.at(right, 236.0f), s.at(right + 13.0f, 249.0f), arrows_.uv(17, 14),
                   arrows_.uv(30, 27));
}

// Without the game's art (another release, or DATA.BIN unreadable), a plain
// screen with the same choices.
void Launcher::draw_plain(ImDrawList *draw, const Screen &s) const {
    draw->AddRectFilled({0, 0}, s.window, IM_COL32(222, 206, 176, 255));
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.3f);
    for (int i = 0; i < static_cast<int>(Item::Count); ++i) {
        const ImVec2 at = s.at(74.0f, kFirstRow + kRowPitch * static_cast<float>(i));
        if (i == row_) draw->AddRectFilled({0, at.y - 6.0f * s.scale}, s.at(239.0f, kFirstRow + kRowPitch * i + 24.0f),
                                           IM_COL32(250, 246, 236, 255));
        draw->AddText(at, kInk, kLabels[i]);
    }
    ImGui::PopFont();
    if (asking_) draw->AddText(s.at(120.0f, 230.0f), kInk, answer_yes_ ? "Quit?  > Yes    No" : "Quit?    Yes  > No");
}

void Launcher::handle_input(const Screen &s) {
    Layer &layer = Layer::get();
    const bool back = layer.take_back() ||
                      ImGui::IsKeyPressed(layer.confirm_south() ? ImGuiKey_GamepadFaceRight : ImGuiKey_GamepadFaceDown,
                                          false);
    const bool confirm =
        ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false) ||
        ImGui::IsKeyPressed(ImGuiKey_Space, false) ||
        (layer.gamepad_armed() &&
         ImGui::IsKeyPressed(layer.confirm_south() ? ImGuiKey_GamepadFaceDown : ImGuiKey_GamepadFaceRight, false));
    const bool up = ImGui::IsKeyPressed(ImGuiKey_UpArrow) || ImGui::IsKeyPressed(ImGuiKey_GamepadDpadUp) ||
                    ImGui::IsKeyPressed(ImGuiKey_GamepadLStickUp);
    const bool down = ImGui::IsKeyPressed(ImGuiKey_DownArrow) || ImGui::IsKeyPressed(ImGuiKey_GamepadDpadDown) ||
                      ImGui::IsKeyPressed(ImGuiKey_GamepadLStickDown);
    const bool sideways = ImGui::IsKeyPressed(ImGuiKey_LeftArrow) || ImGui::IsKeyPressed(ImGuiKey_RightArrow) ||
                          ImGui::IsKeyPressed(ImGuiKey_GamepadDpadLeft) ||
                          ImGui::IsKeyPressed(ImGuiKey_GamepadDpadRight) ||
                          ImGui::IsKeyPressed(ImGuiKey_GamepadLStickLeft) ||
                          ImGui::IsKeyPressed(ImGuiKey_GamepadLStickRight);
    const ImGuiIO &io = ImGui::GetIO();
    const bool moved = io.MousePos.x != last_mouse_.x || io.MousePos.y != last_mouse_.y;
    last_mouse_ = io.MousePos;
    (void)layer.take_dropped_file();

    if (asking_) {
        if (sideways) {
            answer_yes_ = !answer_yes_;
            mixer_.play(cursor_sound_);
        }
        if (back) {
            asking_ = false;
            mixer_.play(cancel_sound_);
        }
        if (confirm || ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                // A click on Yes or No answers with it.
                const float x = (io.MousePos.x - s.origin.x) / s.scale;
                const float y = (io.MousePos.y - s.origin.y) / s.scale;
                if (y < 232.0f || y > 252.0f || x < 186.0f || x > 294.0f) return;
                answer_yes_ = x < 240.0f;
            }
            asking_ = false;
            mixer_.play(decide_sound_);
            if (answer_yes_) leave(LauncherChoice::Quit);
        }
        return;
    }
    const int count = static_cast<int>(Item::Count);
    const int before = row_;
    if (up) row_ = (row_ + count - 1) % count;
    if (down) row_ = (row_ + 1) % count;
    // The mouse: a row under the pointer is chosen, a click opens it.
    const float mx = (io.MousePos.x - s.origin.x) / s.scale;
    const float my = (io.MousePos.y - s.origin.y) / s.scale;
    std::optional<int> hovered;
    for (int i = 0; i < count; ++i) {
        const float y = kFirstRow + kRowPitch * static_cast<float>(i);
        if (mx >= 36.0f && mx < 240.0f && my >= y - 8.0f && my < y + 24.0f) hovered = i;
    }
    if (hovered && moved) row_ = *hovered;
    if (row_ != before) mixer_.play(cursor_sound_);
    if (confirm || (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left))) {
        mixer_.play(decide_sound_);
        activate(static_cast<Item>(row_));
    } else if (ImGui::IsKeyPressed(ImGuiKey_GamepadStart, false)) {
        mixer_.play(decide_sound_);
        leave(LauncherChoice::Play);
    } else if (layer.take_menu_toggle()) {
        mixer_.play(decide_sound_);
        activate(Item::Settings);
    } else if (back) {
        mixer_.play(cancel_sound_);
        asking_ = true;
        answer_yes_ = false;
    }
}

bool Launcher::frame() {
    const double now = ImGui::GetTime();
    if (!started_) started_ = now;
    const float fade_in = ease(static_cast<float>((now - *started_) / kFadeSeconds));
    const float fade_out = leaving_since_ ? ease(static_cast<float>((now - *leaving_since_) / kFadeSeconds)) : 0.0f;
    const float music_in = ease(static_cast<float>((now - *started_) / kMusicFadeInSeconds));
    mixer_.pump(kMusicLevel * music_in * (1.0f - fade_out));

    const Screen s = Screen::of(ImGui::GetIO().DisplaySize);
    ImDrawList *draw = ImGui::GetBackgroundDrawList();
    if (art_ready()) {
        draw_background(draw, s);
        draw_title(draw, s, "MAIN MENU");
        draw_cart(draw, s);
        draw_rows(draw, s);
        draw_footer(draw, s);
        if (asking_) draw_question(draw, s, "Do you want to quit?");
    } else {
        draw_plain(draw, s);
    }

    if (launcher_menu_open()) {
        if (!launcher_menu_frame() && take_launcher_menu_quit()) {
            // Quit, set up again or restart, chosen in the settings; main
            // acts on the last two.
            choice_ = LauncherChoice::Quit;
            return false;
        }
    } else if (!leaving_since_) {
        handle_input(s);
    }

    // In from black, and out to black into the game, which comes up from
    // black as well.
    const float black = std::max(1.0f - fade_in, fade_out);
    if (black > 0.001f)
        ImGui::GetForegroundDrawList()->AddRectFilled({0, 0}, s.window, with_alpha(IM_COL32_BLACK, black));
    // Gone once the screen is black and the decide sound has rung out.
    return !(leaving_since_ && now - *leaving_since_ >= kFadeSeconds &&
             (!mixer_.busy() || now - *leaving_since_ >= 2.0));
}

// Whether this start shows the launcher: the setting, but not for scripted
// runs (their frame numbers count from the game's start) unless
// MHP3RD_LAUNCHER asks for it, and not right after the setup, whose last
// screen already says Play.
bool wanted(bool after_setup) {
    if (install::take_launcher_skip()) return false;
    if (after_setup || !Layer::get().attached()) return false;
    if (settings::overridden_by("ui.launcher") == nullptr &&
        (std::getenv("MHP3RD_INPUT_SCRIPT") != nullptr || std::getenv("MHP3RD_INPUT_LIVE") != nullptr ||
         std::getenv("MHP3RD_AUTO_CONFIRM") != nullptr))
        return false;
    return settings::current().launcher;
}

} // namespace

LauncherChoice run_launcher(const fs::path &disc_image, bool after_setup) {
    if (!wanted(after_setup)) return LauncherChoice::Play;
    Layer &layer = Layer::get();
    layer.set_interactive(true);
    layer.renderer().set_game_input(false);
    Launcher launcher(disc_image);
    const bool window_open = layer.run([&] { return launcher.frame(); }, false);
    layer.renderer().set_game_input(true);
    layer.set_interactive(false);
    return window_open ? launcher.choice() : LauncherChoice::Quit;
}

} // namespace mhp3rd::ui
