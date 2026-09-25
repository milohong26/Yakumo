#include "gpu/post_process.hpp"

#include <algorithm>
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

    std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
    for (std::uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
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
        !make_pass(bloom_format_, false, pass_rgba16f_, error) || !make_pass(kSceneColor, true, pass_target_, error))
        return false;
    if (!make_pipeline(kPostDepthShader, sizeof(kPostDepthShader), pass_r32f_, depth_pipeline_, error) ||
        !make_pipeline(kPostAoShader, sizeof(kPostAoShader), pass_rg16f_, ao_pipeline_, error) ||
        !make_pipeline(kPostBlurShader, sizeof(kPostBlurShader), pass_rg16f_, blur_pipeline_, error) ||
        !make_pipeline(kPostBloomDownShader, sizeof(kPostBloomDownShader), pass_rgba16f_, down_pipeline_, error) ||
        !make_pipeline(kPostBloomUpShader, sizeof(kPostBloomUpShader), pass_rgba16f_, up_pipeline_, error) ||
        !make_pipeline(kPostCompositeShader, sizeof(kPostCompositeShader), pass_target_, composite_pipeline_, error))
        return false;
    ready_ = true;
    return true;
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

VkDescriptorSet Effects::make_set(std::array<VkImageView, 5> views, std::array<bool, 5> linear) {
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool_;
    allocate.descriptorSetCount = 1u;
    allocate.pSetLayouts = &set_layout_;
    VkDescriptorSet set{};
    if (vkAllocateDescriptorSets(device_, &allocate, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    std::array<VkDescriptorImageInfo, 5> images{};
    std::array<VkWriteDescriptorSet, 5> writes{};
    for (std::uint32_t i = 0; i < 5u; ++i) {
        // Bindings a pass does not use still get a valid image.
        const VkImageView view = views[i] != VK_NULL_HANDLE ? views[i] : distances_.view;
        images[i] = {linear[i] ? linear_ : nearest_, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1u;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &images[i];
    }
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
        !make_image(visibility_b_, half, kVisibility, kDrawn, VK_IMAGE_ASPECT_COLOR_BIT, pass_rg16f_, error)) {
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
    const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5u * 32u};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = 32u;
    pool.poolSizeCount = 1u;
    pool.pPoolSizes = &size;
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
    composite_set_ = make_set({scene_color_.view, scene_depth_.view, distances_.view, visibility_b_.view, up_[0].view},
                              {L, N, N, L, L});
    sized_ = true;
    primed_ = false;
    return true;
}

// The first time after a resize: every image the effects draw starts as
// "nothing occluded, no bloom" in the layout the passes sample it in.
void Effects::prime(VkCommandBuffer commands) {
    if (primed_) return;
    primed_ = true;
    std::vector<std::pair<Image *, float>> images{{&distances_, 1.0e6f}, {&visibility_a_, 1.0f}, {&visibility_b_, 1.0f}};
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
    for (Image &image : down_) destroy_image(image);
    for (Image &image : up_) destroy_image(image);
}

void Effects::destroy() {
    if (device_ == VK_NULL_HANDLE) return;
    destroy_sized();
    for (VkPipeline pipeline :
         {depth_pipeline_, ao_pipeline_, blur_pipeline_, down_pipeline_, up_pipeline_, composite_pipeline_})
        vkDestroyPipeline(device_, pipeline, nullptr);
    for (VkRenderPass pass : {pass_r32f_, pass_rg16f_, pass_rgba16f_, pass_target_})
        vkDestroyRenderPass(device_, pass, nullptr);
    vkDestroyShaderModule(device_, vertex_, nullptr);
    vkDestroyPipelineLayout(device_, layout_, nullptr);
    vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    vkDestroySampler(device_, linear_, nullptr);
    vkDestroySampler(device_, nearest_, nullptr);
    if (timer_ != VK_NULL_HANDLE) vkDestroyQueryPool(device_, timer_, nullptr);
    timer_ = VK_NULL_HANDLE;
    timer_next_ = 0u;
    depth_pipeline_ = ao_pipeline_ = blur_pipeline_ = down_pipeline_ = up_pipeline_ = composite_pipeline_ = {};
    pass_r32f_ = pass_rg16f_ = pass_rgba16f_ = pass_target_ = {};
    vertex_ = {};
    layout_ = {};
    set_layout_ = {};
    linear_ = nearest_ = {};
    ready_ = false;
    device_ = VK_NULL_HANDLE;
}

void Effects::forget(VkImageView color_view) {
    const auto found = target_framebuffers_.find(color_view);
    if (found == target_framebuffers_.end()) return;
    vkDestroyFramebuffer(device_, found->second, nullptr);
    target_framebuffers_.erase(found);
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
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0u, 1u, &set, 0u, nullptr);
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

    stamp(commands, slot, 1u);
    // Depth to half-resolution view distance.
    run(commands, pass_r32f_, distances_.framebuffer, distances_.extent, depth_pipeline_, depth_set_, params);
    stamp(commands, slot, 2u);

    const bool occlusion = options.ambient_occlusion > 0.0f || (options.contact_shadows > 0.0f && camera.light_strength > 0.0f);
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
    }
    if (!occlusion) stamp(commands, slot, 3u);
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
    composite.c[3] = 0.0f;
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
    if (text == nullptr) return options;
    const std::string list = std::string(text) + ",";
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
    }
    return options;
}

} // namespace mhp3rd::gpu::post
