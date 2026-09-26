#include "gpu/post_process.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace mhp3rd::gpu::post {
namespace {

#include "post_shaders.inc"

constexpr VkFormat kSceneColor = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDistance = VK_FORMAT_R32_SFLOAT;
constexpr VkFormat kVisibility = VK_FORMAT_R16G16_SFLOAT;


bool check(VkResult result, const char *what, std::string &error) {
    if (result == VK_SUCCESS) return true;
    error = std::string(what) + " failed (" + std::to_string(static_cast<int>(result)) + ")";
    return false;
}

void barrier(VkCommandBuffer commands, VkImage image, VkImageLayout from, VkImageLayout to, VkImageAspectFlags aspect,
             VkPipelineStageFlags source_stage, VkAccessFlags source_access, VkPipelineStageFlags destination_stage,
             VkAccessFlags destination_access) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0u, 1u, 0u, 1u};
    b.srcAccessMask = source_access;
    b.dstAccessMask = destination_access;
    vkCmdPipelineBarrier(commands, source_stage, destination_stage, 0u, 0u, nullptr, 0u, nullptr, 1u, &b);
}

VkExtent2D half_of(VkExtent2D extent, int levels) {
    return {std::max(1u, extent.width >> levels), std::max(1u, extent.height >> levels)};
}

// Column-major 4x4 matrices, as the GE's: m[column * 4 + row].
using Matrix = std::array<float, 16>;
using Vector = std::array<float, 3>;

Matrix multiply(const Matrix &a, const Matrix &b) {
    Matrix result{};
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) sum += a[k * 4 + row] * b[column * 4 + k];
            result[column * 4 + row] = sum;
        }
    return result;
}

// The inverse of an affine matrix (the GE's view matrices are 4x3).
Matrix inverse_affine(const Matrix &m) {
    const auto at = [&](int row, int column) { return m[column * 4 + row]; };
    const float c00 = at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1);
    const float c01 = at(1, 2) * at(2, 0) - at(1, 0) * at(2, 2);
    const float c02 = at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0);
    const float det = at(0, 0) * c00 + at(0, 1) * c01 + at(0, 2) * c02;
    Matrix r{};
    if (std::fabs(det) < 1e-20f) {
        r[0] = r[5] = r[10] = r[15] = 1.0f;
        return r;
    }
    const float inv = 1.0f / det;
    // The transposed cofactors over the determinant, placed as columns.
    const float i00 = c00 * inv, i10 = c01 * inv, i20 = c02 * inv;
    const float i01 = (at(0, 2) * at(2, 1) - at(0, 1) * at(2, 2)) * inv;
    const float i11 = (at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0)) * inv;
    const float i21 = (at(0, 1) * at(2, 0) - at(0, 0) * at(2, 1)) * inv;
    const float i02 = (at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1)) * inv;
    const float i12 = (at(0, 2) * at(1, 0) - at(0, 0) * at(1, 2)) * inv;
    const float i22 = (at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0)) * inv;
    const float tx = at(0, 3), ty = at(1, 3), tz = at(2, 3);
    r = {i00, i10, i20, 0.0f, i01, i11, i21, 0.0f, i02, i12, i22, 0.0f,
         -(i00 * tx + i01 * ty + i02 * tz), -(i10 * tx + i11 * ty + i12 * tz), -(i20 * tx + i21 * ty + i22 * tz), 1.0f};
    return r;
}

Vector normalized(Vector v) {
    const float length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (length < 1e-12f) return {0.0f, 1.0f, 0.0f};
    return {v[0] / length, v[1] / length, v[2] / length};
}
Vector cross(const Vector &a, const Vector &b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
float dot(const Vector &a, const Vector &b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// The direction towards the sun in the game's world, whose up axis is +y:
// the game's own where it is known, else the options' angles.
Vector sun_direction(const Options &options);
Vector sun_direction(const Camera &camera, const Options &options) {
    if (options.sun_from_game && camera.sun_known) return normalized(camera.sun);
    return sun_direction(options);
}
Vector sun_direction(const Options &options) {
    constexpr float kDegrees = 3.14159265f / 180.0f;
    const float elevation = std::clamp(options.sun_elevation, 5.0f, 89.0f) * kDegrees;
    const float azimuth = options.sun_azimuth * kDegrees;
    return normalized({std::cos(elevation) * std::sin(azimuth), std::sin(elevation),
                       std::cos(elevation) * std::cos(azimuth)});
}

// The Sun block of post_sun.glsl (std140).
struct SunBlock {
    float view_to_shadow[16];
    float direction[4];
    float color[4];
    float shade[4];
    float params[4];
    float rays[4];
    float view_to_world[16];
    float water[4];
    float clouds[4];
};

} // namespace

// The push constants every pass shares (post_common.glsl).
struct Effects::Params {
    float proj[4];
    float depth[4];
    float range[4];
    float viewport[4];
    float light[4];
    float a[4];
    float b[4];
    float c[4];
};
static_assert(sizeof(float) * 32 == 128, "push constants are 128 bytes");

std::uint32_t Effects::memory_type(std::uint32_t bits, VkMemoryPropertyFlags flags) const {
    for (std::uint32_t i = 0; i < memory_.memoryTypeCount; ++i)
        if ((bits & (1u << i)) != 0u && (memory_.memoryTypes[i].propertyFlags & flags) == flags) return i;
    return 0u;
}

bool Effects::make_pass(VkFormat format, bool keep_target, VkRenderPass &pass, std::string &error) {
    VkAttachmentDescription attachment{};
    attachment.format = format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    // Every pixel is written, so nothing is loaded.
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // The game's target stays in its attachment layout; the effects' own
    // images end up ready to be sampled by the next pass.
    attachment.initialLayout = keep_target ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout =
        keep_target ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAttachmentReference reference{0u, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1u;
    subpass.pColorAttachments = &reference;
    // Before: what earlier passes wrote and read of it (and the copies of the
    // scene) is done. After: the next pass samples it, or the game draws on
    // and copies the target.
    std::array<VkSubpassDependency, 2> dependencies{};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0u;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].srcSubpass = 0u;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    info.attachmentCount = 1u;
    info.pAttachments = &attachment;
    info.subpassCount = 1u;
    info.pSubpasses = &subpass;
    info.dependencyCount = static_cast<std::uint32_t>(dependencies.size());
    info.pDependencies = dependencies.data();
    return check(vkCreateRenderPass(device_, &info, nullptr, &pass), "vkCreateRenderPass (effects)", error);
}

bool Effects::make_pipeline(const std::uint32_t *fragment, std::size_t bytes, VkRenderPass pass,
                            VkPipeline &pipeline, std::string &error) {
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = bytes;
    module_info.pCode = fragment;
    VkShaderModule module{};
    if (!check(vkCreateShaderModule(device_, &module_info, nullptr, &module), "vkCreateShaderModule (effects)", error))
        return false;
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex_;
    stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = module;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1u;
    viewport.scissorCount = 1u;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blending.attachmentCount = 1u;
    blending.pAttachments = &blend;
    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.stageCount = static_cast<std::uint32_t>(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pColorBlendState = &blending;
    info.pDynamicState = &dynamic;
    info.layout = layout_;
    info.renderPass = pass;
    const bool made = check(vkCreateGraphicsPipelines(device_, cache_, 1u, &info, nullptr, &pipeline),
                            "vkCreateGraphicsPipelines (effects)", error);
    vkDestroyShaderModule(device_, module, nullptr);
    return made;
}

bool Effects::create(VkDevice device, VkPhysicalDevice physical_device, VkPipelineCache cache, std::string &error) {
    destroy();
    device_ = device;
    cache_ = cache;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_);
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    timestamp_period_ = properties.limits.timestampPeriod;
    sun_stride_ = std::max<VkDeviceSize>(
        256u, (sizeof(SunBlock) + properties.limits.minUniformBufferOffsetAlignment - 1u) /
                  properties.limits.minUniformBufferOffsetAlignment * properties.limits.minUniformBufferOffsetAlignment);
    // Timed only for MHP3RD_TRACE_EFFECTS: on Metal each timestamp splits the
    // work into another encoder, which costs more than the stage it times.
    if (properties.limits.timestampComputeAndGraphics && std::getenv("MHP3RD_TRACE_EFFECTS") != nullptr) {
        VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query.queryCount = kTimerSlots * kStamps;
        if (vkCreateQueryPool(device_, &query, nullptr, &timer_) != VK_SUCCESS) timer_ = VK_NULL_HANDLE;
    }
    // Bloom in packed floats where the GPU renders to them (half the
    // bandwidth of 16-bit channels), else 16-bit floats.
    VkFormatProperties packed{};
    vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_B10G11R11_UFLOAT_PACK32, &packed);
    constexpr VkFormatFeatureFlags kNeeded = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                             VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                             VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    bloom_format_ = (packed.optimalTilingFeatures & kNeeded) == kNeeded ? VK_FORMAT_B10G11R11_UFLOAT_PACK32
                                                                        : VK_FORMAT_R16G16B16A16_SFLOAT;

    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.maxLod = 0.0f;
    if (!check(vkCreateSampler(device_, &sampler, nullptr, &linear_), "vkCreateSampler (effects)", error)) return false;
    sampler.magFilter = VK_FILTER_NEAREST;
    sampler.minFilter = VK_FILTER_NEAREST;
    if (!check(vkCreateSampler(device_, &sampler, nullptr, &nearest_), "vkCreateSampler (effects)", error))
        return false;

    // Depth compared against the shadow map, filtered between texels; outside
    // it everything is lit.
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.compareEnable = VK_TRUE;
    sampler.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    if (!check(vkCreateSampler(device_, &sampler, nullptr, &shadow_sampler_), "vkCreateSampler (effects)", error))
        return false;

    // Bindings 0-4, 7, 8, 10 and 11: the pass's images; 5: the shadow map,
    // compared; 6: the sun; 9: the shadow map's depths.
    std::array<VkDescriptorSetLayoutBinding, 12> bindings{};
    for (std::uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType =
            i == 6u ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1u;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    set_info.pBindings = bindings.data();
    if (!check(vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &set_layout_),
               "vkCreateDescriptorSetLayout (effects)", error))
        return false;
    VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0u, sizeof(Params)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1u;
    layout_info.pSetLayouts = &set_layout_;
    layout_info.pushConstantRangeCount = 1u;
    layout_info.pPushConstantRanges = &push;
    if (!check(vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout_), "vkCreatePipelineLayout (effects)",
               error))
        return false;

    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = sizeof(kPostVertexShader);
    module_info.pCode = kPostVertexShader;
    if (!check(vkCreateShaderModule(device_, &module_info, nullptr, &vertex_), "vkCreateShaderModule (effects)", error))
        return false;
    if (!make_pass(kDistance, false, pass_r32f_, error) || !make_pass(kVisibility, false, pass_rg16f_, error) ||
        !make_pass(bloom_format_, false, pass_rgba16f_, error) || !make_pass(kSceneColor, true, pass_target_, error) ||
        !make_pass(VK_FORMAT_R16G16B16A16_SFLOAT, false, pass_average_, error))
        return false;
    if (!make_pipeline(kPostDepthShader, sizeof(kPostDepthShader), pass_r32f_, depth_pipeline_, error) ||
        !make_pipeline(kPostAoShader, sizeof(kPostAoShader), pass_rg16f_, ao_pipeline_, error) ||
        !make_pipeline(kPostBlurShader, sizeof(kPostBlurShader), pass_rg16f_, blur_pipeline_, error) ||
        !make_pipeline(kPostBloomDownShader, sizeof(kPostBloomDownShader), pass_rgba16f_, down_pipeline_, error) ||
        !make_pipeline(kPostBloomUpShader, sizeof(kPostBloomUpShader), pass_rgba16f_, up_pipeline_, error) ||
        !make_pipeline(kPostCompositeShader, sizeof(kPostCompositeShader), pass_target_, composite_pipeline_, error) ||
        !make_pipeline(kPostRaysShader, sizeof(kPostRaysShader), pass_rg16f_, rays_pipeline_, error) ||
        !make_pipeline(kPostAverageShader, sizeof(kPostAverageShader), pass_average_, average_pipeline_, error) ||
        !make_pipeline(kPostReflectShader, sizeof(kPostReflectShader), pass_average_, reflect_pipeline_, error))
        return false;
    if (!make_shadow_resources(error)) return false;
    ready_ = true;
    return true;
}

bool Effects::make_shadow_resources(std::string &error) {
    // The shadow map and the pass that draws it: cleared to the far plane,
    // left ready to be sampled.
    VkAttachmentDescription attachment{};
    attachment.format = VK_FORMAT_D32_SFLOAT;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    VkAttachmentReference reference{0u, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pDepthStencilAttachment = &reference;
    // After the passes of earlier frames that sampled the map; before the
    // passes that sample it now.
    std::array<VkSubpassDependency, 2> dependencies{};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0u;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask =
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask =
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].srcSubpass = 0u;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass.attachmentCount = 1u;
    pass.pAttachments = &attachment;
    pass.subpassCount = 1u;
    pass.pSubpasses = &subpass;
    pass.dependencyCount = static_cast<std::uint32_t>(dependencies.size());
    pass.pDependencies = dependencies.data();
    if (!check(vkCreateRenderPass(device_, &pass, nullptr, &shadow_pass_), "vkCreateRenderPass (shadows)", error))
        return false;
    if (!make_image(shadow_, {kShadowSize, kShadowSize}, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT, shadow_pass_, error))
        return false;

    // The sun's blocks, written by the CPU for each record().
    VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer.size = sun_stride_ * kSunSlots;
    buffer.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!check(vkCreateBuffer(device_, &buffer, nullptr, &sun_buffer_), "vkCreateBuffer (sun)", error)) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, sun_buffer_, &requirements);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!check(vkAllocateMemory(device_, &allocate, nullptr, &sun_memory_), "vkAllocateMemory (sun)", error))
        return false;
    vkBindBufferMemory(device_, sun_buffer_, sun_memory_, 0u);
    if (!check(vkMapMemory(device_, sun_memory_, 0u, VK_WHOLE_SIZE, 0u, &sun_mapped_), "vkMapMemory (sun)", error))
        return false;
    std::memset(sun_mapped_, 0, static_cast<std::size_t>(buffer.size));
    return true;
}

bool Effects::make_shadow_pipeline(VkPipelineLayout layout, std::uint32_t stride, std::uint32_t position_offset,
                                   std::uint32_t texcoord_offset, std::string &error) {
    if (shadow_pass_ == VK_NULL_HANDLE) return false;
    std::array<VkShaderModule, 2> modules{};
    const std::array<std::pair<const std::uint32_t *, std::size_t>, 2> code{
        std::pair{kShadowVertexShader, sizeof(kShadowVertexShader)},
        std::pair{kShadowFragmentShader, sizeof(kShadowFragmentShader)}};
    for (std::size_t i = 0; i < modules.size(); ++i) {
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = code[i].second;
        info.pCode = code[i].first;
        if (!check(vkCreateShaderModule(device_, &info, nullptr, &modules[i]), "vkCreateShaderModule (shadows)",
                   error)) {
            for (VkShaderModule module : modules) vkDestroyShaderModule(device_, module, nullptr);
            return false;
        }
    }
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    for (std::size_t i = 0; i < stages.size(); ++i) {
        stages[i] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stages[i].stage = i == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[i].module = modules[i];
        stages[i].pName = "main";
    }
    const VkVertexInputBindingDescription binding{0u, stride, VK_VERTEX_INPUT_RATE_VERTEX};
    const std::array<VkVertexInputAttributeDescription, 2> attributes{
        VkVertexInputAttributeDescription{0u, 0u, VK_FORMAT_R32G32B32A32_SFLOAT, position_offset},
        VkVertexInputAttributeDescription{1u, 0u, VK_FORMAT_R32G32_SFLOAT, texcoord_offset}};
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = 1u;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions = attributes.data();
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1u;
    viewport.scissorCount = 1u;
    // Both faces cast (the game's thin walls and cloth are single-sided);
    // a slope-scaled bias keeps a surface from shadowing itself.
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.depthBiasEnable = VK_TRUE;
    raster.depthBiasConstantFactor = 2.0f;
    raster.depthBiasSlopeFactor = 2.5f;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.stageCount = static_cast<std::uint32_t>(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blending;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    info.renderPass = shadow_pass_;
    const bool made = check(vkCreateGraphicsPipelines(device_, cache_, 1u, &info, nullptr, &shadow_pipeline_),
                            "vkCreateGraphicsPipelines (shadows)", error);
    for (VkShaderModule module : modules) vkDestroyShaderModule(device_, module, nullptr);
    return made;
}

bool Effects::make_water_pipeline(VkPipelineLayout layout, VkFormat depth_format, std::uint32_t stride,
                                  std::uint32_t position_offset, std::uint32_t color_offset, std::string &error) {
    // The mask, cleared, and the target's depth, tested and kept as it is.
    std::array<VkAttachmentDescription, 2> attachments{};
    attachments[0].format = VK_FORMAT_R8_UNORM;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    attachments[1].format = depth_format;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    const VkAttachmentReference color_reference{0u, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    const VkAttachmentReference depth_reference{1u, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1u;
    subpass.pColorAttachments = &color_reference;
    subpass.pDepthStencilAttachment = &depth_reference;
    constexpr VkPipelineStageFlags kTests =
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    std::array<VkSubpassDependency, 2> dependencies{};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0u;
    dependencies[0].srcStageMask = kTests | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = kTests | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].srcSubpass = 0u;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = kTests | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = kTests | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_TRANSFER_READ_BIT;
    VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass.attachmentCount = static_cast<std::uint32_t>(attachments.size());
    pass.pAttachments = attachments.data();
    pass.subpassCount = 1u;
    pass.pSubpasses = &subpass;
    pass.dependencyCount = static_cast<std::uint32_t>(dependencies.size());
    pass.pDependencies = dependencies.data();
    if (!check(vkCreateRenderPass(device_, &pass, nullptr, &water_pass_), "vkCreateRenderPass (water)", error))
        return false;

    std::array<VkShaderModule, 2> modules{};
    const std::array<std::pair<const std::uint32_t *, std::size_t>, 2> code{
        std::pair{kWaterVertexShader, sizeof(kWaterVertexShader)},
        std::pair{kWaterFragmentShader, sizeof(kWaterFragmentShader)}};
    for (std::size_t i = 0; i < modules.size(); ++i) {
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = code[i].second;
        info.pCode = code[i].first;
        if (!check(vkCreateShaderModule(device_, &info, nullptr, &modules[i]), "vkCreateShaderModule (water)",
                   error)) {
            for (VkShaderModule module : modules) vkDestroyShaderModule(device_, module, nullptr);
            return false;
        }
    }
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    for (std::size_t i = 0; i < stages.size(); ++i) {
        stages[i] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stages[i].stage = i == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[i].module = modules[i];
        stages[i].pName = "main";
    }
    const VkVertexInputBindingDescription binding{0u, stride, VK_VERTEX_INPUT_RATE_VERTEX};
    const std::array<VkVertexInputAttributeDescription, 2> attributes{
        VkVertexInputAttributeDescription{0u, 0u, VK_FORMAT_R32G32B32A32_SFLOAT, position_offset},
        VkVertexInputAttributeDescription{2u, 0u, VK_FORMAT_R8G8B8A8_UNORM, color_offset}};
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = 1u;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions = attributes.data();
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1u;
    viewport.scissorCount = 1u;
    // The water wrote this depth itself; a nudge towards the camera lets it
    // pass its own test whatever the two shaders' rounding.
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.depthBiasEnable = VK_TRUE;
    raster.depthBiasConstantFactor = -8.0f;
    raster.depthBiasSlopeFactor = -1.0f;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_FALSE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    // Overlapping pieces keep the most water.
    blend.blendEnable = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.colorBlendOp = VK_BLEND_OP_MAX;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.alphaBlendOp = VK_BLEND_OP_MAX;
    VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blending.attachmentCount = 1u;
    blending.pAttachments = &blend;
    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.stageCount = static_cast<std::uint32_t>(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blending;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    info.renderPass = water_pass_;
    const bool made = check(vkCreateGraphicsPipelines(device_, cache_, 1u, &info, nullptr, &water_pipeline_),
                            "vkCreateGraphicsPipelines (water)", error);
    for (VkShaderModule module : modules) vkDestroyShaderModule(device_, module, nullptr);
    return made;
}

bool Effects::begin_water(VkCommandBuffer commands, VkImageView color_view, VkImageView depth_view) {
    if (!ready() || water_pipeline_ == VK_NULL_HANDLE || depth_view == VK_NULL_HANDLE) return false;
    VkFramebuffer &framebuffer = water_framebuffers_[color_view];
    if (framebuffer == VK_NULL_HANDLE) {
        const std::array<VkImageView, 2> views{water_mask_.view, depth_view};
        VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        info.renderPass = water_pass_;
        info.attachmentCount = static_cast<std::uint32_t>(views.size());
        info.pAttachments = views.data();
        info.width = extent_.width;
        info.height = extent_.height;
        info.layers = 1u;
        if (vkCreateFramebuffer(device_, &info, nullptr, &framebuffer) != VK_SUCCESS) {
            water_framebuffers_.erase(color_view);
            return false;
        }
    }
    std::array<VkClearValue, 2> clears{};
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = water_pass_;
    begin.framebuffer = framebuffer;
    begin.renderArea = {{0, 0}, extent_};
    begin.clearValueCount = static_cast<std::uint32_t>(clears.size());
    begin.pClearValues = clears.data();
    vkCmdBeginRenderPass(commands, &begin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, water_pipeline_);
    return true;
}

void Effects::end_water(VkCommandBuffer commands) {
    vkCmdEndRenderPass(commands);
    water_this_frame_ = true;
}

bool Effects::begin_shadows(VkCommandBuffer commands, const Camera &camera, const Options &options,
                            Matrix &world_to_clip) {
    if (!ready_ || shadow_pipeline_ == VK_NULL_HANDLE || !camera.valid || options.sun <= 0.0f) return false;
    // The camera: where it is and where it looks, in the world.
    const Matrix inverse_view = inverse_affine(camera.view);
    const Vector eye{inverse_view[12], inverse_view[13], inverse_view[14]};
    const Vector forward = normalized({-inverse_view[8], -inverse_view[9], -inverse_view[10]});
    const float range = std::max(options.shadow_range, 100.0f);
    const Vector centre{eye[0] + forward[0] * range * 0.55f, eye[1] + forward[1] * range * 0.55f,
                        eye[2] + forward[2] * range * 0.55f};

    // The sun looks along -sun; a rotation only, so the box can be snapped
    // to whole texels and the shadows' edges do not crawl as the camera moves.
    const Vector to_sun = sun_direction(camera, options);
    const Vector look{-to_sun[0], -to_sun[1], -to_sun[2]};
    const Vector up_hint = std::fabs(to_sun[1]) > 0.99f ? Vector{0.0f, 0.0f, 1.0f} : Vector{0.0f, 1.0f, 0.0f};
    const Vector side = normalized(cross(look, up_hint));
    const Vector up = cross(side, look);
    Matrix light{side[0], up[0], -look[0], 0.0f, side[1], up[1], -look[1], 0.0f,
                 side[2], up[2], -look[2], 0.0f, 0.0f,    0.0f,  0.0f,     1.0f};
    const float texel = 2.0f * range / static_cast<float>(kShadowSize);
    const float cx = std::floor(dot(side, centre) / texel) * texel;
    const float cy = std::floor(dot(up, centre) / texel) * texel;
    const float cz = -dot(look, centre);  // light space z (the camera looks down -z)
    // Casters up to this far towards the sun, and receivers as far beyond.
    constexpr float kDepth = 6000.0f;
    const float left = cx - range, right = cx + range, bottom = cy - range, top = cy + range;
    const float near_plane = -(cz + kDepth), far_plane = -(cz - kDepth);
    Matrix ortho{};
    ortho[0] = 2.0f / (right - left);
    ortho[5] = 2.0f / (top - bottom);
    ortho[10] = -1.0f / (far_plane - near_plane);
    ortho[12] = -(right + left) / (right - left);
    ortho[13] = -(top + bottom) / (top - bottom);
    ortho[14] = -near_plane / (far_plane - near_plane);
    ortho[15] = 1.0f;
    world_to_clip = multiply(ortho, light);
    shadow_world_to_clip_ = world_to_clip;
    shadow_texel_world_ = texel;

    VkClearValue clear{};
    clear.depthStencil = {1.0f, 0u};
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = shadow_pass_;
    begin.framebuffer = shadow_.framebuffer;
    begin.renderArea = {{0, 0}, {kShadowSize, kShadowSize}};
    begin.clearValueCount = 1u;
    begin.pClearValues = &clear;
    vkCmdBeginRenderPass(commands, &begin, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0.0f, 0.0f, static_cast<float>(kShadowSize), static_cast<float>(kShadowSize), 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, {kShadowSize, kShadowSize}};
    vkCmdSetViewport(commands, 0u, 1u, &viewport);
    vkCmdSetScissor(commands, 0u, 1u, &scissor);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);
    return true;
}

void Effects::end_shadows(VkCommandBuffer commands) {
    vkCmdEndRenderPass(commands);
    shadow_primed_ = true;
    shadow_this_frame_ = true;
}

void Effects::write_sun(const Camera &camera, const Options &options) {
    SunBlock block{};
    const Vector to_sun = sun_direction(camera, options);
    const Matrix &view = camera.view;
    const Vector in_view = normalized({view[0] * to_sun[0] + view[4] * to_sun[1] + view[8] * to_sun[2],
                                       view[1] * to_sun[0] + view[5] * to_sun[1] + view[9] * to_sun[2],
                                       view[2] * to_sun[0] + view[6] * to_sun[1] + view[10] * to_sun[2]});
    // Clip space to shadow map coordinates: x and y from [-1, 1] to [0, 1].
    Matrix to_map{};
    to_map[0] = 0.5f;
    to_map[5] = 0.5f;
    to_map[10] = 1.0f;
    to_map[12] = 0.5f;
    to_map[13] = 0.5f;
    to_map[15] = 1.0f;
    const Matrix view_to_shadow = multiply(to_map, multiply(shadow_world_to_clip_, inverse_affine(view)));
    std::memcpy(block.view_to_shadow, view_to_shadow.data(), sizeof(block.view_to_shadow));
    block.direction[0] = in_view[0];
    block.direction[1] = in_view[1];
    block.direction[2] = in_view[2];
    block.direction[3] = shadow_drawn_ ? 1.0f : 0.0f;
    // Late-morning sun and the blue of the sky in the shade.
    // The sun takes the hue and strength of the game's own key light: a
    // dim, blue light at night makes weak, blue moonlight.
    const float w = options.warmth;
    std::array<float, 3> key{1.0f, 1.0f, 1.0f};
    float strength = 1.0f;
    if (options.sun_from_game && camera.sun_known) {
        const std::array<float, 3> &c = camera.sun_color;
        const float luma = std::max(0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2], 0.01f);
        for (int i = 0; i < 3; ++i) key[i] = 1.0f + (c[i] / luma - 1.0f) * 0.6f;
        strength = std::clamp(luma / 0.7f, 0.25f, 1.15f);
    }
    block.color[0] = key[0];
    block.color[1] = key[1] * (1.0f - 0.1f * w);
    block.color[2] = key[2] * (1.0f - 0.26f * w);
    block.color[3] = options.sun * strength;
    block.shade[0] = 1.0f - 0.18f * w;
    block.shade[1] = 1.0f - 0.1f * w;
    block.shade[2] = 1.0f + 0.08f * w;
    block.shade[3] = options.shade;
    block.params[0] = 1.0f / static_cast<float>(kShadowSize);
    block.params[1] = shadow_texel_world_;
    block.params[2] = options.sun > 0.0f ? 1.0f : 0.0f;
    block.params[3] = options.shadow_range;
    block.rays[0] = options.rays;
    block.rays[1] = options.rays_reach;
    block.rays[2] = options.rays_g;
    // World units across the shadow map's depth, and how wide the sun's
    // disc makes a penumbra per unit of distance to the caster.
    block.rays[3] = options.softness;
    const Matrix view_to_world = inverse_affine(camera.view);
    std::memcpy(block.view_to_world, view_to_world.data(), sizeof(block.view_to_world));
    static const auto started = std::chrono::steady_clock::now();
    block.water[0] = options.water;
    block.water[1] = options.ripples;
    block.water[2] = std::fmod(std::chrono::duration<float>(std::chrono::steady_clock::now() - started).count(), 3600.0f);
    block.water[3] = water_this_frame_ && options.water > 0.0f ? 1.0f : 0.0f;
    block.clouds[0] = options.clouds;
    block.clouds[1] = std::clamp(options.cloud_cover, 0.0f, 1.0f);
    block.clouds[2] = std::max(options.cloud_size, 100.0f);
    const std::uint32_t slot = sun_next_++ % kSunSlots;
    sun_offset_ = static_cast<std::uint32_t>(slot * sun_stride_);
    std::memcpy(static_cast<std::uint8_t *>(sun_mapped_) + sun_offset_, &block, sizeof(block));
}

bool Effects::make_image(Image &image, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage,
                         VkImageAspectFlags aspect, VkRenderPass pass, std::string &error) {
    image.extent = extent;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {extent.width, extent.height, 1u};
    info.mipLevels = 1u;
    info.arrayLayers = 1u;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!check(vkCreateImage(device_, &info, nullptr, &image.image), "vkCreateImage (effects)", error)) return false;
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device_, image.image, &requirements);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!check(vkAllocateMemory(device_, &allocate, nullptr, &image.memory), "vkAllocateMemory (effects)", error))
        return false;
    vkBindImageMemory(device_, image.image, image.memory, 0u);
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = image.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange = {aspect, 0u, 1u, 0u, 1u};
    if (!check(vkCreateImageView(device_, &view, nullptr, &image.view), "vkCreateImageView (effects)", error))
        return false;
    if (pass == VK_NULL_HANDLE) return true;
    VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer.renderPass = pass;
    framebuffer.attachmentCount = 1u;
    framebuffer.pAttachments = &image.view;
    framebuffer.width = extent.width;
    framebuffer.height = extent.height;
    framebuffer.layers = 1u;
    return check(vkCreateFramebuffer(device_, &framebuffer, nullptr, &image.framebuffer),
                 "vkCreateFramebuffer (effects)", error);
}

void Effects::destroy_image(Image &image) {
    if (device_ == VK_NULL_HANDLE) return;
    vkDestroyFramebuffer(device_, image.framebuffer, nullptr);
    vkDestroyImageView(device_, image.view, nullptr);
    vkDestroyImage(device_, image.image, nullptr);
    vkFreeMemory(device_, image.memory, nullptr);
    image = Image{};
}

VkDescriptorSet Effects::make_set(std::array<VkImageView, kImageBindings> views,
                                  std::array<bool, kImageBindings> linear) {
    constexpr std::array<std::uint32_t, kImageBindings> kBinding{0u, 1u, 2u, 3u, 4u, 7u, 8u, 10u, 11u};
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool_;
    allocate.descriptorSetCount = 1u;
    allocate.pSetLayouts = &set_layout_;
    VkDescriptorSet set{};
    if (vkAllocateDescriptorSets(device_, &allocate, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    std::array<VkDescriptorImageInfo, kImageBindings + 2u> images{};
    std::array<VkWriteDescriptorSet, kImageBindings + 3u> writes{};
    const auto write_image = [&](std::size_t i, std::uint32_t binding, VkDescriptorImageInfo info) {
        images[i] = info;
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set;
        writes[i].dstBinding = binding;
        writes[i].descriptorCount = 1u;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &images[i];
    };
    for (std::size_t i = 0; i < kImageBindings; ++i) {
        // Bindings a pass does not use still get a valid image.
        const VkImageView view = views[i] != VK_NULL_HANDLE ? views[i] : distances_.view;
        write_image(i, kBinding[i], {linear[i] ? linear_ : nearest_, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
    }
    // The shadow map, compared (5) and as depths (9).
    write_image(kImageBindings, 5u, {shadow_sampler_, shadow_.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL});
    write_image(kImageBindings + 1u, 9u, {nearest_, shadow_.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL});
    const VkDescriptorBufferInfo sun{sun_buffer_, 0u, sizeof(SunBlock)};
    VkWriteDescriptorSet &sun_write = writes[kImageBindings + 2u];
    sun_write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    sun_write.dstSet = set;
    sun_write.dstBinding = 6u;
    sun_write.descriptorCount = 1u;
    sun_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    sun_write.pBufferInfo = &sun;
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);
    return set;
}

bool Effects::resize(VkExtent2D extent, VkFormat depth_format, std::string &error) {
    if (!ready_) return false;
    destroy_sized();
    extent_ = extent;
    depth_format_ = depth_format;
    const VkExtent2D half = half_of(extent, 1);
    constexpr VkImageUsageFlags kCopied = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // Cleared once, so an effect that is off leaves a defined image behind.
    constexpr VkImageUsageFlags kDrawn =
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const VkImageAspectFlags depth_only = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (!make_image(scene_color_, extent, kSceneColor, kCopied, VK_IMAGE_ASPECT_COLOR_BIT, VK_NULL_HANDLE, error) ||
        !make_image(scene_depth_, extent, depth_format, kCopied, depth_only, VK_NULL_HANDLE, error) ||
        !make_image(distances_, half, kDistance, kDrawn, VK_IMAGE_ASPECT_COLOR_BIT, pass_r32f_, error) ||
        !make_image(visibility_a_, half, kVisibility, kDrawn, VK_IMAGE_ASPECT_COLOR_BIT, pass_rg16f_, error) ||
        !make_image(visibility_b_, half, kVisibility, kDrawn, VK_IMAGE_ASPECT_COLOR_BIT, pass_rg16f_, error) ||
        !make_image(rays_, half_of(extent, 2), kVisibility, kDrawn, VK_IMAGE_ASPECT_COLOR_BIT, pass_rg16f_, error) ||
        !make_image(average_, {1u, 1u}, VK_FORMAT_R16G16B16A16_SFLOAT, kDrawn, VK_IMAGE_ASPECT_COLOR_BIT,
                    pass_average_, error) ||
        !make_image(water_mask_, extent, VK_FORMAT_R8_UNORM, kDrawn, VK_IMAGE_ASPECT_COLOR_BIT, VK_NULL_HANDLE,
                    error) ||
        !make_image(reflection_, half, VK_FORMAT_R16G16B16A16_SFLOAT, kDrawn, VK_IMAGE_ASPECT_COLOR_BIT,
                    pass_average_, error)) {
        destroy_sized();
        return false;
    }
    for (int i = 0; i < kBloomLevels; ++i) {
        const VkExtent2D level = half_of(extent, i + 2);
        if (!make_image(down_[i], level, bloom_format_, kDrawn, VK_IMAGE_ASPECT_COLOR_BIT, pass_rgba16f_, error) ||
            !make_image(up_[i], level, bloom_format_, kDrawn, VK_IMAGE_ASPECT_COLOR_BIT, pass_rgba16f_, error)) {
            destroy_sized();
            return false;
        }
    }
    const std::array<VkDescriptorPoolSize, 2> sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 11u * 32u},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 32u}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = 32u;
    pool.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    pool.pPoolSizes = sizes.data();
    if (!check(vkCreateDescriptorPool(device_, &pool, nullptr, &pool_), "vkCreateDescriptorPool (effects)", error)) {
        destroy_sized();
        return false;
    }
    constexpr bool L = true;
    constexpr bool N = false;
    depth_set_ = make_set({scene_depth_.view}, {N});
    ao_set_ = make_set({distances_.view}, {N});
    blur_set_ = make_set({visibility_a_.view, distances_.view}, {N, N});
    for (int i = 0; i < kBloomLevels; ++i) {
        down_sets_[i] = make_set({i == 0 ? scene_color_.view : down_[i - 1].view}, {L});
        up_sets_[i] = make_set({i == kBloomLevels - 1 ? down_[i].view : up_[i + 1].view, down_[i].view}, {L, L});
    }
    composite_set_ = make_set({scene_color_.view, scene_depth_.view, distances_.view, visibility_b_.view, up_[0].view,
                               rays_.view, average_.view, water_mask_.view, reflection_.view},
                              {L, N, N, L, L, L, N, N, L});
    rays_set_ = make_set({distances_.view}, {N});
    average_set_ = make_set({visibility_b_.view, distances_.view, scene_color_.view}, {N, N, N});
    reflect_set_ = make_set({distances_.view, scene_color_.view, water_mask_.view}, {N, L, L});
    sized_ = true;
    primed_ = false;
    return true;
}

// The first time after a resize: every image the effects draw starts as
// "nothing occluded, no bloom" in the layout the passes sample it in.
void Effects::prime(VkCommandBuffer commands) {
    if (!shadow_primed_) {
        // Nothing drawn into the shadow map yet: all of it at the far plane.
        shadow_primed_ = true;
        barrier(commands, shadow_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0u, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkClearDepthStencilValue far{1.0f, 0u};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0u, 1u, 0u, 1u};
        vkCmdClearDepthStencilImage(commands, shadow_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &far, 1u, &range);
        barrier(commands, shadow_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT);
    }
    if (primed_) return;
    primed_ = true;
    std::vector<std::pair<Image *, float>> images{
        {&distances_, 1.0e6f}, {&visibility_a_, 1.0f}, {&visibility_b_, 1.0f}, {&rays_, 0.0f}, {&average_, 1.0f},
        {&water_mask_, 0.0f},  {&reflection_, 0.0f}};
    for (Image &image : down_) images.push_back({&image, 0.0f});
    for (Image &image : up_) images.push_back({&image, 0.0f});
    for (const auto &[image, value] : images) {
        barrier(commands, image->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0u, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkClearColorValue clear{{value, value, value, value}};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
        vkCmdClearColorImage(commands, image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1u, &range);
        barrier(commands, image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
}

void Effects::destroy_sized() {
    sized_ = false;
    if (device_ == VK_NULL_HANDLE) return;
    for (auto &[view, framebuffer] : target_framebuffers_) vkDestroyFramebuffer(device_, framebuffer, nullptr);
    target_framebuffers_.clear();
    if (pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
    destroy_image(scene_color_);
    destroy_image(scene_depth_);
    destroy_image(distances_);
    destroy_image(visibility_a_);
    destroy_image(visibility_b_);
    destroy_image(rays_);
    destroy_image(average_);
    destroy_image(water_mask_);
    destroy_image(reflection_);
    for (auto &[view, framebuffer] : water_framebuffers_) vkDestroyFramebuffer(device_, framebuffer, nullptr);
    water_framebuffers_.clear();
    for (Image &image : down_) destroy_image(image);
    for (Image &image : up_) destroy_image(image);
}

void Effects::destroy() {
    if (device_ == VK_NULL_HANDLE) return;
    destroy_sized();
    for (VkPipeline pipeline :
         {depth_pipeline_, ao_pipeline_, blur_pipeline_, down_pipeline_, up_pipeline_, composite_pipeline_,
          rays_pipeline_, average_pipeline_, reflect_pipeline_, water_pipeline_})
        vkDestroyPipeline(device_, pipeline, nullptr);
    for (VkRenderPass pass : {pass_r32f_, pass_rg16f_, pass_rgba16f_, pass_target_, pass_average_})
        vkDestroyRenderPass(device_, pass, nullptr);
    destroy_image(shadow_);
    vkDestroyPipeline(device_, shadow_pipeline_, nullptr);
    vkDestroyRenderPass(device_, shadow_pass_, nullptr);
    vkDestroySampler(device_, shadow_sampler_, nullptr);
    if (sun_mapped_ != nullptr) vkUnmapMemory(device_, sun_memory_);
    vkDestroyBuffer(device_, sun_buffer_, nullptr);
    vkFreeMemory(device_, sun_memory_, nullptr);
    shadow_pipeline_ = VK_NULL_HANDLE;
    shadow_pass_ = VK_NULL_HANDLE;
    shadow_sampler_ = VK_NULL_HANDLE;
    sun_buffer_ = VK_NULL_HANDLE;
    sun_memory_ = VK_NULL_HANDLE;
    sun_mapped_ = nullptr;
    shadow_primed_ = shadow_drawn_ = shadow_this_frame_ = false;
    vkDestroyShaderModule(device_, vertex_, nullptr);
    vkDestroyPipelineLayout(device_, layout_, nullptr);
    vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    vkDestroySampler(device_, linear_, nullptr);
    vkDestroySampler(device_, nearest_, nullptr);
    if (timer_ != VK_NULL_HANDLE) vkDestroyQueryPool(device_, timer_, nullptr);
    timer_ = VK_NULL_HANDLE;
    timer_next_ = 0u;
    depth_pipeline_ = ao_pipeline_ = blur_pipeline_ = down_pipeline_ = up_pipeline_ = composite_pipeline_ = {};
    rays_pipeline_ = average_pipeline_ = reflect_pipeline_ = water_pipeline_ = VK_NULL_HANDLE;
    vkDestroyRenderPass(device_, water_pass_, nullptr);
    water_pass_ = VK_NULL_HANDLE;
    pass_r32f_ = pass_rg16f_ = pass_rgba16f_ = pass_target_ = pass_average_ = {};
    vertex_ = {};
    layout_ = {};
    set_layout_ = {};
    linear_ = nearest_ = {};
    ready_ = false;
    device_ = VK_NULL_HANDLE;
}

void Effects::forget(VkImageView color_view) {
    for (auto *framebuffers : {&target_framebuffers_, &water_framebuffers_}) {
        const auto found = framebuffers->find(color_view);
        if (found == framebuffers->end()) continue;
        vkDestroyFramebuffer(device_, found->second, nullptr);
        framebuffers->erase(found);
    }
}

void Effects::run(VkCommandBuffer commands, VkRenderPass pass, VkFramebuffer framebuffer, VkExtent2D extent,
                  VkPipeline pipeline, VkDescriptorSet set, const Params &params) {
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = pass;
    begin.framebuffer = framebuffer;
    begin.renderArea = {{0, 0}, extent};
    vkCmdBeginRenderPass(commands, &begin, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height),
                              0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, extent};
    vkCmdSetViewport(commands, 0u, 1u, &viewport);
    vkCmdSetScissor(commands, 0u, 1u, &scissor);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0u, 1u, &set, 1u, &sun_offset_);
    vkCmdPushConstants(commands, layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0u, sizeof(Params), &params);
    vkCmdDraw(commands, 3u, 1u, 0u, 0u);
    vkCmdEndRenderPass(commands);
}

void Effects::record(VkCommandBuffer commands, VkImage color, VkImageView color_view, VkImage depth,
                     VkImageAspectFlags depth_aspect, const Camera &camera, const Options &options, bool bloom) {
    if (!ready() || !camera.valid) return;
    std::uint32_t slot = 0u;
    if (timer_ != VK_NULL_HANDLE) {
        slot = static_cast<std::uint32_t>(timer_next_ % kTimerSlots);
        // This slot was written a lap ago; that frame is long done.
        if (timer_next_ >= kTimerSlots) read_timer(slot);
        ++timer_next_;
        vkCmdResetQueryPool(commands, timer_, slot * kStamps, kStamps);
    }
    stamp(commands, slot, 0u);
    prime(commands);
    VkFramebuffer &target_framebuffer = target_framebuffers_[color_view];
    if (target_framebuffer == VK_NULL_HANDLE) {
        VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        info.renderPass = pass_target_;
        info.attachmentCount = 1u;
        info.pAttachments = &color_view;
        info.width = extent_.width;
        info.height = extent_.height;
        info.layers = 1u;
        if (vkCreateFramebuffer(device_, &info, nullptr, &target_framebuffer) != VK_SUCCESS) {
            target_framebuffers_.erase(color_view);
            return;
        }
    }

    // The scene as it is now: a copy of its colour and depth to read from
    // while the result goes into the target.
    constexpr VkPipelineStageFlags kDrawing =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    constexpr VkAccessFlags kDrawn = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier(commands, color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, kDrawing, kDrawn, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    barrier(commands, depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            depth_aspect, kDrawing, kDrawn, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    barrier(commands, scene_color_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    barrier(commands, scene_depth_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            depth_aspect, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy copy{};
    copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u};
    copy.dstSubresource = copy.srcSubresource;
    copy.extent = {extent_.width, extent_.height, 1u};
    vkCmdCopyImage(commands, color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, scene_color_.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1u, &copy);
    copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vkCmdCopyImage(commands, depth, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, scene_depth_.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1u, &copy);
    barrier(commands, color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    barrier(commands, depth, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            depth_aspect, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    barrier(commands, scene_color_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    barrier(commands, scene_depth_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, depth_aspect, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

    Params params{};
    const std::array<float, 16> &m = camera.projection;
    params.proj[0] = m[0];
    params.proj[1] = m[5];
    params.proj[2] = m[8];
    params.proj[3] = m[9];
    params.depth[0] = m[10];
    params.depth[1] = m[14];
    params.depth[2] = m[11];
    params.range[0] = camera.depth_min;
    params.range[1] = camera.depth_max;
    params.range[2] = camera.viewport_x;
    params.range[3] = camera.viewport_y;
    params.viewport[0] = camera.viewport_width;
    params.viewport[1] = camera.viewport_height;
    params.viewport[2] = static_cast<float>(extent_.width);
    params.viewport[3] = static_cast<float>(extent_.height);
    params.light[0] = camera.light[0];
    params.light[1] = camera.light[1];
    params.light[2] = camera.light[2];
    params.light[3] = options.contact_shadows > 0.0f ? camera.light_strength : 0.0f;
    // A game frame's own record() takes the shadow map it just drew, or none;
    // the frames interpolated after it keep that one.
    if (bloom) {
        shadow_drawn_ = shadow_this_frame_;
        shadow_this_frame_ = false;
    }
    write_sun(camera, options);
    if (options.sun > 0.0f) {
        // Contact shadows, when on, look towards the sun.
        SunBlock *block = reinterpret_cast<SunBlock *>(static_cast<std::uint8_t *>(sun_mapped_) + sun_offset_);
        params.light[0] = block->direction[0];
        params.light[1] = block->direction[1];
        params.light[2] = block->direction[2];
        params.light[3] = options.contact_shadows > 0.0f ? 1.0f : 0.0f;
    }

    stamp(commands, slot, 1u);
    // Depth to half-resolution view distance.
    run(commands, pass_r32f_, distances_.framebuffer, distances_.extent, depth_pipeline_, depth_set_, params);
    stamp(commands, slot, 2u);

    const bool occlusion = options.ambient_occlusion > 0.0f || options.sun > 0.0f ||
                           (options.contact_shadows > 0.0f && camera.light_strength > 0.0f);
    if (occlusion) {
        Params ao = params;
        ao.a[0] = options.occlusion_radius;
        ao.a[1] = 0.615f;
        // Target pixels a view unit covers at distance 1.
        ao.a[2] = std::abs(m[5] * camera.viewport_height) * 0.5f;
        ao.a[3] = 0.0f;
        ao.b[0] = options.shadow_length;
        ao.b[1] = options.shadow_thickness;
        ao.b[2] = options.contact_shadows > 0.0f ? 12.0f : 0.0f;
        ao.b[3] = options.shadow_distance;
        run(commands, pass_rg16f_, visibility_a_.framebuffer, visibility_a_.extent, ao_pipeline_, ao_set_, ao);
        stamp(commands, slot, 3u);
        run(commands, pass_rg16f_, visibility_b_.framebuffer, visibility_b_.extent, blur_pipeline_, blur_set_, params);
        // The frames interpolated between the game's keep the game frame's
        // average: it changes slowly.
        if (options.sun > 0.0f && bloom)
            run(commands, pass_average_, average_.framebuffer, average_.extent, average_pipeline_, average_set_, params);
    }
    if (water_this_frame_ && options.water > 0.0f)
        run(commands, pass_average_, reflection_.framebuffer, reflection_.extent, reflect_pipeline_, reflect_set_,
            params);
    water_this_frame_ = false;
    if (!occlusion) stamp(commands, slot, 3u);
    // Light shafts are soft and slow: the frames interpolated between the
    // game's keep the game frame's.
    if (options.rays > 0.0f && options.sun > 0.0f && shadow_drawn_ && bloom) {
        Params rays = params;
        rays.a[0] = 12.0f;
        rays.a[1] = static_cast<float>(sun_next_ % 64u) * 7.0f;
        run(commands, pass_rg16f_, rays_.framebuffer, rays_.extent, rays_pipeline_, rays_set_, rays);
    }
    stamp(commands, slot, 4u);

    if (options.bloom > 0.0f && bloom) {
        for (int i = 0; i < kBloomLevels; ++i) {
            const VkExtent2D source = i == 0 ? extent_ : down_[i - 1].extent;
            Params down = params;
            down.a[0] = i == 0 ? 1.0f : 0.0f;
            down.a[1] = options.bloom_threshold;
            down.a[2] = options.bloom_knee;
            // The first level takes a quarter of the scene: taps twice as far.
            const float spread = i == 0 ? 2.0f : 1.0f;
            down.a[3] = spread / static_cast<float>(source.width);
            down.b[0] = spread / static_cast<float>(source.height);
            down.b[1] = options.shoulder;
            down.b[2] = options.bloom_cap;
            run(commands, pass_rgba16f_, down_[i].framebuffer, down_[i].extent, down_pipeline_, down_sets_[i], down);
        }
        for (int i = kBloomLevels - 1; i >= 0; --i) {
            const VkExtent2D smaller = i == kBloomLevels - 1 ? down_[i].extent : up_[i + 1].extent;
            Params up = params;
            up.a[0] = 1.0f / static_cast<float>(smaller.width);
            up.a[1] = 1.0f / static_cast<float>(smaller.height);
            // The smallest level is its own source: it adds nothing to itself.
            up.a[2] = i == kBloomLevels - 1 ? 0.0f : 1.0f;
            run(commands, pass_rgba16f_, up_[i].framebuffer, up_[i].extent, up_pipeline_, up_sets_[i], up);
        }
    }

    stamp(commands, slot, 5u);
    Params composite = params;
    composite.a[0] = occlusion ? options.ambient_occlusion : 0.0f;
    composite.a[1] = occlusion && camera.light_strength > 0.0f ? options.contact_shadows : 0.0f;
    // The levels were added up without normalising: six of them.
    composite.a[2] = options.bloom / static_cast<float>(kBloomLevels);
    composite.a[3] = options.sharpen;
    composite.b[0] = options.exposure;
    composite.b[1] = options.contrast;
    composite.b[2] = options.saturation;
    composite.b[3] = options.vibrance;
    composite.c[0] = options.vignette;
    composite.c[1] = options.split_toning;
    composite.c[2] = options.shoulder;
    composite.c[3] = options.antialias;
    composite.light[0] = camera.fog_end;
    composite.light[1] = camera.fog_scale;
    composite.light[2] = camera.fog ? 1.0f : 0.0f;
    composite.light[3] = static_cast<float>(options.debug);
    if (!occlusion) {
        // Nothing wrote the visibility this frame: the composite must not
        // apply what an earlier frame left in it.
        composite.a[0] = 0.0f;
        composite.a[1] = 0.0f;
    }
    run(commands, pass_target_, target_framebuffer, extent_, composite_pipeline_, composite_set_, composite);
    stamp(commands, slot, 6u);
}

void Effects::stamp(VkCommandBuffer commands, std::uint32_t slot, std::uint32_t index) {
    if (timer_ == VK_NULL_HANDLE) return;
    vkCmdWriteTimestamp(commands, index == 0u ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        timer_, slot * kStamps + index);
}

void Effects::read_timer(std::uint32_t slot) {
    std::array<std::uint64_t, kStamps> stamps{};
    if (vkGetQueryPoolResults(device_, timer_, slot * kStamps, kStamps, sizeof(stamps), stamps.data(),
                              sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        return;
    if (stamps[kStamps - 1u] <= stamps[0]) return;
    const double to_ms = timestamp_period_ * 1e-6;
    timer_sum_ms_ += static_cast<double>(stamps[kStamps - 1u] - stamps[0]) * to_ms;
    for (std::uint32_t i = 0; i + 1u < kStamps; ++i)
        stage_sum_ms_[i] += static_cast<double>(stamps[i + 1u] >= stamps[i] ? stamps[i + 1u] - stamps[i] : 0u) * to_ms;
    ++timer_count_;
}

double Effects::take_gpu_ms(std::array<double, 6> *stages) {
    const double ms = timer_count_ != 0u ? timer_sum_ms_ / timer_count_ : -1.0;
    if (stages != nullptr)
        for (std::size_t i = 0; i < stages->size(); ++i)
            (*stages)[i] = timer_count_ != 0u ? stage_sum_ms_[i] / timer_count_ : 0.0;
    timer_sum_ms_ = 0.0;
    stage_sum_ms_ = {};
    timer_count_ = 0u;
    return ms;
}

Options options_from_environment(Options options) {
    const char *text = std::getenv("MHP3RD_EFFECTS_OPTIONS");
    return text != nullptr ? options_from_text(options, text) : options;
}

Options options_from_text(Options options, const std::string &text) {
    std::string list = text + ",";
    for (char &c : list)
        if (c == '\n' || c == '\r' || c == ';' || c == ' ') c = ',';
    std::size_t start = 0u;
    for (std::size_t comma = list.find(','); comma != std::string::npos; comma = list.find(',', start)) {
        const std::string item = list.substr(start, comma - start);
        start = comma + 1u;
        const std::size_t equals = item.find('=');
        if (equals == std::string::npos) continue;
        const std::string name = item.substr(0, equals);
        const float value = std::strtof(item.c_str() + equals + 1u, nullptr);
        if (name == "ao") options.ambient_occlusion = value;
        else if (name == "radius") options.occlusion_radius = value;
        else if (name == "shadows") options.contact_shadows = value;
        else if (name == "bloom") options.bloom = value;
        else if (name == "threshold") options.bloom_threshold = value;
        else if (name == "knee") options.bloom_knee = value;
        else if (name == "cap") options.bloom_cap = value;
        else if (name == "sharpen") options.sharpen = value;
        else if (name == "exposure") options.exposure = value;
        else if (name == "contrast") options.contrast = value;
        else if (name == "saturation") options.saturation = value;
        else if (name == "vibrance") options.vibrance = value;
        else if (name == "vignette") options.vignette = value;
        else if (name == "split") options.split_toning = value;
        else if (name == "shoulder") options.shoulder = value;
        else if (name == "highlight") options.highlight = value;
        else if (name == "gloss") options.gloss = value;
        else if (name == "rim") options.rim = value;
        else if (name == "wrap") options.wrap = value;
        else if (name == "ground") options.ground = value;
        else if (name == "knee") options.knee = value;
        else if (name == "aa") options.antialias = value;
        else if (name == "sun") options.sun = value;
        else if (name == "shade") options.shade = value;
        else if (name == "elevation") options.sun_elevation = value;
        else if (name == "azimuth") options.sun_azimuth = value;
        else if (name == "range") options.shadow_range = value;
        else if (name == "rays") options.rays = value;
        else if (name == "warmth") options.warmth = value;
        else if (name == "g") options.rays_g = value;
        else if (name == "gamesun") options.sun_from_game = value > 0.5f;
        else if (name == "soft") options.softness = value;
        else if (name == "water") options.water = value;
        else if (name == "ripples") options.ripples = value;
        else if (name == "clouds") options.clouds = value;
        else if (name == "cover") options.cloud_cover = value;
        else if (name == "cloudsize") options.cloud_size = value;
        else if (name == "reach") options.rays_reach = value;
        else if (name == "debug") options.debug = static_cast<int>(value);
    }
    return options;
}

} // namespace mhp3rd::gpu::post
