#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <map>
#include <string>

// Lighting and image effects on the game's finished 3D scene, before its
// interface is drawn over it: ground-truth ambient occlusion and contact
// shadows reconstructed from the depth buffer, bloom, sharpening and a film
// grade. The scene is read from a target (its colour and depth) and the
// result written back into its colour, so everything that later uses the
// picture (the interface drawn over it, frame interpolation's pictures, the
// write-back to guest memory) sees it.
//
// Every image, pass and pipeline is made up front (and again when the target
// size changes), so the effects never compile a pipeline while playing.
namespace mhp3rd::gpu::post {

// The camera the scene was drawn with, as its draws into the displayed
// framebuffer set it: enough to turn depth back into view-space positions.
struct Camera {
    bool valid{};
    std::array<float, 16> projection{};  // column major, as the GE's
    float viewport_x{};                  // the Vulkan viewport, in target pixels
    float viewport_y{};
    float viewport_width{};
    float viewport_height{};  // negative: flipped, as the GE's
    float depth_min{};
    float depth_max{};
    std::array<float, 3> light{};  // view-space direction towards the key light
    float light_strength{};        // 0: no key light
    bool fog{};                    // the scene is drawn with the GE's fog
    float fog_end{};
    float fog_scale{};
};

// How strong each effect is. Zero turns one off.
struct Options {
    float ambient_occlusion{0.85f};
    float occlusion_radius{42.0f};  // in the game's units: a hunter is about 170 tall
    float contact_shadows{0.0f};  // off: without temporal filtering the rays are too noisy yet
    float shadow_length{36.0f};
    float shadow_thickness{24.0f};
    float shadow_distance{1500.0f};
    // Bloom works on light above the shoulder's knee, expanded as if the
    // picture had not been clipped to white (up to bloom_cap).
    float bloom{0.12f};
    float bloom_threshold{1.0f};
    float bloom_knee{0.5f};
    float bloom_cap{3.0f};
    float sharpen{0.2f};
    float exposure{1.0f};
    float contrast{1.06f};
    float saturation{1.03f};
    float vibrance{0.15f};
    float vignette{0.14f};
    float split_toning{0.6f};
    float shoulder{0.8f};
    // Lit models shaded per pixel (video.lighting; ge.frag): the strength of
    // the lights' highlights and their sharpness (Blinn-Phong power), the rim
    // light, how far diffuse light wraps past the terminator, and the ambient
    // light from below, against 2 - ground from above; light past the knee
    // rolls off towards white instead of clipping (0 clips, as the GE).
    float highlight{0.8f};
    float gloss{24.0f};
    float rim{0.45f};
    float wrap{0.2f};
    float ground{0.75f};
    float knee{0.6f};
    int debug{};  // 1 occlusion, 2 contact shadows, 3 distance, 4 bloom
};

class Effects {
public:
    Effects() = default;
    Effects(const Effects &) = delete;
    Effects &operator=(const Effects &) = delete;
    ~Effects() { destroy(); }

    bool create(VkDevice device, VkPhysicalDevice physical_device, VkPipelineCache cache, std::string &error);
    void destroy();

    // Makes the images for targets of `extent` whose depth has `depth_format`.
    // The caller waits for the device first when it replaces them.
    bool resize(VkExtent2D extent, VkFormat depth_format, std::string &error);
    [[nodiscard]] bool ready() const noexcept { return ready_ && sized_; }
    [[nodiscard]] VkExtent2D extent() const noexcept { return extent_; }

    // Records the effects on the scene in `color` and `depth`, a target in its
    // attachment layouts, outside a render pass; `color` gets the result and
    // both are back in their attachment layouts afterwards.
    // `bloom`: make the bloom anew; else the last one is used again (for the
    // frames interpolated between two of the game's, where it cannot differ
    // visibly).
    void record(VkCommandBuffer commands, VkImage color, VkImageView color_view, VkImage depth,
                VkImageAspectFlags depth_aspect, const Camera &camera, const Options &options, bool bloom = true);

    // A target's colour view that is going away: its framebuffer goes too.
    void forget(VkImageView color_view);

    // GPU time the effects took, averaged over the presents since the last
    // call, in milliseconds; negative when nothing was measured. `stages`
    // gets the time of each stage: copies, depth, occlusion, denoise, bloom,
    // composite.
    [[nodiscard]] double take_gpu_ms(std::array<double, 6> *stages = nullptr);

private:
    struct Image {
        VkImage image{};
        VkDeviceMemory memory{};
        VkImageView view{};
        VkFramebuffer framebuffer{};
        VkExtent2D extent{};
    };
    struct Params;

    bool make_image(Image &image, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage,
                    VkImageAspectFlags aspect, VkRenderPass pass, std::string &error);
    void destroy_image(Image &image);
    void destroy_sized();
    void prime(VkCommandBuffer commands);
    bool make_pass(VkFormat format, bool keep_target, VkRenderPass &pass, std::string &error);
    bool make_pipeline(const std::uint32_t *fragment, std::size_t bytes, VkRenderPass pass, VkPipeline &pipeline,
                       std::string &error);
    VkDescriptorSet make_set(std::array<VkImageView, 5> views, std::array<bool, 5> linear);
    void run(VkCommandBuffer commands, VkRenderPass pass, VkFramebuffer framebuffer, VkExtent2D extent,
             VkPipeline pipeline, VkDescriptorSet set, const Params &params);
    [[nodiscard]] std::uint32_t memory_type(std::uint32_t bits, VkMemoryPropertyFlags flags) const;

    VkDevice device_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    VkPipelineCache cache_{};
    bool ready_{};
    bool sized_{};
    bool primed_{};
    VkExtent2D extent_{};
    VkFormat depth_format_{};
    VkFormat bloom_format_{VK_FORMAT_R16G16B16A16_SFLOAT};

    VkSampler linear_{};
    VkSampler nearest_{};
    VkDescriptorSetLayout set_layout_{};
    VkPipelineLayout layout_{};
    VkShaderModule vertex_{};
    VkRenderPass pass_r32f_{};
    VkRenderPass pass_rg16f_{};
    VkRenderPass pass_rgba16f_{};
    VkRenderPass pass_target_{};
    VkPipeline depth_pipeline_{};
    VkPipeline ao_pipeline_{};
    VkPipeline blur_pipeline_{};
    VkPipeline down_pipeline_{};
    VkPipeline up_pipeline_{};
    VkPipeline composite_pipeline_{};

    // Bloom from a quarter of the target's size down, in three levels.
    static constexpr int kBloomLevels = 3;
    Image scene_color_;
    Image scene_depth_;
    Image distances_;
    Image visibility_a_;
    Image visibility_b_;
    std::array<Image, kBloomLevels> down_{};
    std::array<Image, kBloomLevels> up_{};
    VkDescriptorPool pool_{};
    VkDescriptorSet depth_set_{};
    VkDescriptorSet ao_set_{};
    VkDescriptorSet blur_set_{};
    std::array<VkDescriptorSet, kBloomLevels> down_sets_{};
    std::array<VkDescriptorSet, kBloomLevels> up_sets_{};
    VkDescriptorSet composite_set_{};
    std::map<VkImageView, VkFramebuffer> target_framebuffers_;

    // Timestamps around each record(), read back a lap of the ring later.
    static constexpr std::uint32_t kTimerSlots = 32u;
    static constexpr std::uint32_t kStamps = 7u;
    VkQueryPool timer_{};
    float timestamp_period_{};
    std::uint64_t timer_next_{};
    double timer_sum_ms_{};
    std::array<double, 6> stage_sum_ms_{};
    std::uint32_t timer_count_{};
    void read_timer(std::uint32_t slot);
    void stamp(VkCommandBuffer commands, std::uint32_t slot, std::uint32_t index);
};

// Options with MHP3RD_EFFECTS_OPTIONS applied: `name=value` pairs separated
// by commas (ao, radius, shadows, bloom, threshold, knee, cap, sharpen,
// exposure, contrast, saturation, vibrance, vignette, split, shoulder,
// highlight, gloss, rim, wrap, ground, knee).
[[nodiscard]] Options options_from_environment(Options options);

} // namespace mhp3rd::gpu::post
