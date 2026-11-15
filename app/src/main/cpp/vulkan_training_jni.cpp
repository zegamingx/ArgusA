#include <jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>

#include "adamw_shader_spv.h"
#include "linear_shader_spv.h"
#include "linear_fp16_shader_spv.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <vector>

namespace {

constexpr char kLogTag[] = "ArgusVulkan";
constexpr uint32_t kLocalWorkGroupSize = 64;

struct AdamwPushConstants {
    uint32_t totalElementCount;
    uint32_t segmentOffset;
    uint32_t segmentElementCount;
    float learningRate;
    float beta1;
    float beta2;
    float biasCorrection1;
    float biasCorrection2;
    float epsilon;
    float gradientClipFactor;
    float weightDecay;
};

// The push-constant layout must match the GLSL block exactly.
static_assert(sizeof(AdamwPushConstants) == 44);
static_assert(offsetof(AdamwPushConstants, segmentOffset) == 4);
static_assert(offsetof(AdamwPushConstants, segmentElementCount) == 8);
static_assert(offsetof(AdamwPushConstants, learningRate) == 12);
static_assert(offsetof(AdamwPushConstants, gradientClipFactor) == 36);
static_assert(offsetof(AdamwPushConstants, weightDecay) == 40);

struct LinearPushConstants {
    uint32_t rowCount;
    uint32_t batchCount;
    uint32_t rowsPerBatch;
    uint32_t inputColumns;
    uint32_t outputColumns;
    uint32_t inputOffset;
    uint32_t weightOffset;
    uint32_t outputOffset;
    uint32_t transposeWeights;
};

static_assert(sizeof(LinearPushConstants) == 36);
static_assert(offsetof(LinearPushConstants, transposeWeights) == 32);

struct VulkanRuntime {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties physicalDeviceProperties{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    uint32_t maxComputeWorkGroupCountX = 0;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipeline linearPipeline = VK_NULL_HANDLE;
    VkPipeline fp16LinearPipeline = VK_NULL_HANDLE;

    uint32_t instanceApiVersion = VK_API_VERSION_1_0;
    bool supportsFp16Compute = false;

    VkBuffer storageBuffer = VK_NULL_HANDLE;
    VkDeviceMemory storageMemory = VK_NULL_HANDLE;
    VkDeviceSize storageCapacity = 0;
    void* mappedStorage = nullptr;
    bool hostCoherent = false;
};

struct GroupArrays {
    jfloatArray parameters = nullptr;
    jfloatArray gradients = nullptr;
    jfloatArray firstMoments = nullptr;
    jfloatArray secondMoments = nullptr;
    jsize elementCount = 0;
    size_t packedOffset = 0;
    float weightDecay = 0.0f;
};

struct AdamwSegment {
    uint32_t elementOffset = 0;
    uint32_t elementCount = 0;
    float weightDecay = 0.0f;
};

std::mutex gRuntimeMutex;
VulkanRuntime gRuntime{};

void logError(const char* message, VkResult result = VK_SUCCESS) {
    if (result == VK_SUCCESS) {
        __android_log_print(ANDROID_LOG_WARN, kLogTag, "%s", message);
    } else {
        __android_log_print(
            ANDROID_LOG_WARN,
            kLogTag,
            "%s (VkResult=%d)",
            message,
            static_cast<int>(result)
        );
    }
}

uint32_t calculateDispatchWorkGroupCount(uint64_t invocationCount) {
    const uint32_t deviceGroupLimit = gRuntime.maxComputeWorkGroupCountX;
    if (invocationCount == 0u || deviceGroupLimit == 0u) {
        return 0u;
    }

    const uint64_t requiredGroups =
        1u + (invocationCount - 1u) / kLocalWorkGroupSize;
    return static_cast<uint32_t>(
        std::min<uint64_t>(requiredGroups, deviceGroupLimit)
    );
}

void destroyStorageBuffer() {
    if (gRuntime.device == VK_NULL_HANDLE) {
        return;
    }

    if (gRuntime.mappedStorage != nullptr &&
        gRuntime.storageMemory != VK_NULL_HANDLE) {
        vkUnmapMemory(gRuntime.device, gRuntime.storageMemory);
    }

    if (gRuntime.storageBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(gRuntime.device, gRuntime.storageBuffer, nullptr);
    }

    if (gRuntime.storageMemory != VK_NULL_HANDLE) {
        vkFreeMemory(gRuntime.device, gRuntime.storageMemory, nullptr);
    }

    gRuntime.storageBuffer = VK_NULL_HANDLE;
    gRuntime.storageMemory = VK_NULL_HANDLE;
    gRuntime.storageCapacity = 0;
    gRuntime.mappedStorage = nullptr;
    gRuntime.hostCoherent = false;
}

void destroyRuntime() {
    if (gRuntime.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(gRuntime.device);
        destroyStorageBuffer();

        if (gRuntime.fp16LinearPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(gRuntime.device, gRuntime.fp16LinearPipeline, nullptr);
        }
        if (gRuntime.linearPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(gRuntime.device, gRuntime.linearPipeline, nullptr);
        }
        if (gRuntime.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(gRuntime.device, gRuntime.pipeline, nullptr);
        }
        if (gRuntime.pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(gRuntime.device, gRuntime.pipelineLayout, nullptr);
        }
        if (gRuntime.descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(gRuntime.device, gRuntime.descriptorPool, nullptr);
        }
        if (gRuntime.descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(
                gRuntime.device,
                gRuntime.descriptorSetLayout,
                nullptr
            );
        }
        if (gRuntime.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(gRuntime.device, gRuntime.commandPool, nullptr);
        }

        vkDestroyDevice(gRuntime.device, nullptr);
    }

    if (gRuntime.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(gRuntime.instance, nullptr);
    }

    gRuntime = VulkanRuntime{};
}

bool createStorageBuffer(VkDeviceSize requiredBytes) {
    if (gRuntime.device == VK_NULL_HANDLE ||
        requiredBytes == 0 ||
        requiredBytes > gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange) {
        return false;
    }

    destroyStorageBuffer();

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = requiredBytes;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(
        gRuntime.device,
        &bufferInfo,
        nullptr,
        &gRuntime.storageBuffer
    );
    if (result != VK_SUCCESS) {
        logError("Failed to create the Vulkan storage buffer.", result);
        destroyStorageBuffer();
        return false;
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(
        gRuntime.device,
        gRuntime.storageBuffer,
        &requirements
    );

    uint32_t selectedMemoryType = std::numeric_limits<uint32_t>::max();
    bool selectedMemoryIsCoherent = false;
    uint32_t fallbackMemoryType = std::numeric_limits<uint32_t>::max();

    for (uint32_t index = 0;
         index < gRuntime.memoryProperties.memoryTypeCount;
         ++index) {
        if ((requirements.memoryTypeBits & (1u << index)) == 0u) {
            continue;
        }

        const VkMemoryPropertyFlags flags =
            gRuntime.memoryProperties.memoryTypes[index].propertyFlags;

        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0u) {
            continue;
        }

        if (fallbackMemoryType == std::numeric_limits<uint32_t>::max()) {
            fallbackMemoryType = index;
        }

        if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u) {
            selectedMemoryType = index;
            selectedMemoryIsCoherent = true;
            break;
        }
    }

    if (selectedMemoryType == std::numeric_limits<uint32_t>::max()) {
        selectedMemoryType = fallbackMemoryType;
        selectedMemoryIsCoherent = false;
    }

    if (selectedMemoryType == std::numeric_limits<uint32_t>::max()) {
        logError("No host-visible Vulkan memory type is available.");
        destroyStorageBuffer();
        return false;
    }

    VkMemoryAllocateInfo allocationInfo{};
    allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocationInfo.allocationSize = requirements.size;
    allocationInfo.memoryTypeIndex = selectedMemoryType;

    result = vkAllocateMemory(
        gRuntime.device,
        &allocationInfo,
        nullptr,
        &gRuntime.storageMemory
    );
    if (result != VK_SUCCESS) {
        logError("Failed to allocate Vulkan storage memory.", result);
        destroyStorageBuffer();
        return false;
    }

    result = vkBindBufferMemory(
        gRuntime.device,
        gRuntime.storageBuffer,
        gRuntime.storageMemory,
        0
    );
    if (result != VK_SUCCESS) {
        logError("Failed to bind Vulkan storage memory.", result);
        destroyStorageBuffer();
        return false;
    }

    result = vkMapMemory(
        gRuntime.device,
        gRuntime.storageMemory,
        0,
        VK_WHOLE_SIZE,
        0,
        &gRuntime.mappedStorage
    );
    if (result != VK_SUCCESS || gRuntime.mappedStorage == nullptr) {
        logError("Failed to map Vulkan storage memory.", result);
        destroyStorageBuffer();
        return false;
    }

    gRuntime.storageCapacity = requiredBytes;
    gRuntime.hostCoherent = selectedMemoryIsCoherent;

    VkDescriptorBufferInfo descriptorInfo{};
    descriptorInfo.buffer = gRuntime.storageBuffer;
    descriptorInfo.offset = 0;
    descriptorInfo.range = requiredBytes;

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = gRuntime.descriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &descriptorInfo;

    vkUpdateDescriptorSets(
        gRuntime.device,
        1,
        &descriptorWrite,
        0,
        nullptr
    );

    return true;
}

bool ensureStorageBuffer(VkDeviceSize requiredBytes) {
    if (gRuntime.storageBuffer != VK_NULL_HANDLE &&
        gRuntime.storageCapacity >= requiredBytes) {
        return requiredBytes <=
            gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange;
    }

    return createStorageBuffer(requiredBytes);
}

bool initializeRuntime() {
    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "Argus";
    applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    applicationInfo.pEngineName = "Argus On-Device Trainer";
    applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    uint32_t loaderApiVersion = VK_API_VERSION_1_0;
    const auto enumerateInstanceVersion =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion")
        );
    if (enumerateInstanceVersion != nullptr) {
        uint32_t reportedVersion = VK_API_VERSION_1_0;
        if (enumerateInstanceVersion(&reportedVersion) == VK_SUCCESS) {
            loaderApiVersion = reportedVersion;
        }
    }
    gRuntime.instanceApiVersion = std::min(loaderApiVersion, VK_API_VERSION_1_2);
    applicationInfo.apiVersion = gRuntime.instanceApiVersion;

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &applicationInfo;

    VkResult result = vkCreateInstance(
        &instanceInfo,
        nullptr,
        &gRuntime.instance
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan instance creation failed.", result);
        return false;
    }

    uint32_t physicalDeviceCount = 0;
    result = vkEnumeratePhysicalDevices(
        gRuntime.instance,
        &physicalDeviceCount,
        nullptr
    );
    if (result != VK_SUCCESS || physicalDeviceCount == 0) {
        logError("No Vulkan physical device was enumerated.", result);
        return false;
    }

    std::vector<VkPhysicalDevice> physicalDevices;
    try {
        physicalDevices.resize(physicalDeviceCount);
    } catch (const std::bad_alloc&) {
        logError("Not enough memory to enumerate Vulkan devices.");
        return false;
    }

    result = vkEnumeratePhysicalDevices(
        gRuntime.instance,
        &physicalDeviceCount,
        physicalDevices.data()
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan physical-device enumeration failed.", result);
        return false;
    }

    int bestScore = -1;
    for (VkPhysicalDevice candidate : physicalDevices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);

        int deviceScore = 0;
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            deviceScore = 30;
        } else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            deviceScore = 20;
        } else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
            deviceScore = 10;
        }

        if (deviceScore == 0) {
            continue;
        }

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(
            candidate,
            &queueFamilyCount,
            nullptr
        );
        if (queueFamilyCount == 0) {
            continue;
        }

        std::vector<VkQueueFamilyProperties> queueFamilies;
        try {
            queueFamilies.resize(queueFamilyCount);
        } catch (const std::bad_alloc&) {
            continue;
        }

        vkGetPhysicalDeviceQueueFamilyProperties(
            candidate,
            &queueFamilyCount,
            queueFamilies.data()
        );

        for (uint32_t familyIndex = 0;
             familyIndex < queueFamilyCount;
             ++familyIndex) {
            const VkQueueFamilyProperties& family = queueFamilies[familyIndex];
            if (family.queueCount == 0 ||
                (family.queueFlags & VK_QUEUE_COMPUTE_BIT) == 0u) {
                continue;
            }

            int score = deviceScore;
            if ((family.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0u) {
                score += 2;
            }

            if (score > bestScore) {
                bestScore = score;
                gRuntime.physicalDevice = candidate;
                gRuntime.queueFamilyIndex = familyIndex;
                gRuntime.physicalDeviceProperties = properties;
            }
        }
    }

    if (gRuntime.physicalDevice == VK_NULL_HANDLE) {
        logError("No integrated or discrete Vulkan GPU with a compute queue was found.");
        return false;
    }

    vkGetPhysicalDeviceMemoryProperties(
        gRuntime.physicalDevice,
        &gRuntime.memoryProperties
    );

    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = gRuntime.queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDevice16BitStorageFeatures supportedStorage16{};
    VkPhysicalDeviceShaderFloat16Int8Features supportedShaderFloat16{};
    VkPhysicalDeviceFeatures2 supportedFeatures{};
    if (gRuntime.instanceApiVersion >= VK_API_VERSION_1_1 &&
        gRuntime.physicalDeviceProperties.apiVersion >= VK_API_VERSION_1_2) {
        supportedStorage16.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        supportedStorage16.pNext = &supportedShaderFloat16;
        supportedShaderFloat16.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        supportedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        supportedFeatures.pNext = &supportedStorage16;

        const auto getPhysicalDeviceFeatures2 =
            reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                vkGetInstanceProcAddr(gRuntime.instance, "vkGetPhysicalDeviceFeatures2")
            );
        if (getPhysicalDeviceFeatures2 != nullptr) {
            getPhysicalDeviceFeatures2(gRuntime.physicalDevice, &supportedFeatures);
            gRuntime.supportsFp16Compute =
                supportedStorage16.storageBuffer16BitAccess == VK_TRUE &&
                supportedShaderFloat16.shaderFloat16 == VK_TRUE;
        }
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    VkPhysicalDevice16BitStorageFeatures enabledStorage16{};
    VkPhysicalDeviceShaderFloat16Int8Features enabledShaderFloat16{};
    VkPhysicalDeviceFeatures2 enabledFeatures{};

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    if (gRuntime.supportsFp16Compute) {
        enabledStorage16.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        enabledStorage16.storageBuffer16BitAccess = VK_TRUE;
        enabledStorage16.pNext = &enabledShaderFloat16;
        enabledShaderFloat16.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        enabledShaderFloat16.shaderFloat16 = VK_TRUE;
        enabledFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        enabledFeatures.pNext = &enabledStorage16;
        deviceInfo.pNext = &enabledFeatures;
        deviceInfo.pEnabledFeatures = nullptr;
    } else {
        deviceInfo.pEnabledFeatures = &deviceFeatures;
    }

    result = vkCreateDevice(
        gRuntime.physicalDevice,
        &deviceInfo,
        nullptr,
        &gRuntime.device
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan logical-device creation failed.", result);
        return false;
    }

    vkGetDeviceQueue(
        gRuntime.device,
        gRuntime.queueFamilyIndex,
        0,
        &gRuntime.computeQueue
    );
    if (gRuntime.computeQueue == VK_NULL_HANDLE) {
        logError("Vulkan compute queue is unavailable.");
        return false;
    }

    gRuntime.maxComputeWorkGroupCountX =
        gRuntime.physicalDeviceProperties.limits.maxComputeWorkGroupCount[0];

    VkCommandPoolCreateInfo commandPoolInfo{};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolInfo.queueFamilyIndex = gRuntime.queueFamilyIndex;

    result = vkCreateCommandPool(
        gRuntime.device,
        &commandPoolInfo,
        nullptr,
        &gRuntime.commandPool
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan command-pool creation failed.", result);
        return false;
    }

    VkCommandBufferAllocateInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferInfo.commandPool = gRuntime.commandPool;
    commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferInfo.commandBufferCount = 1;

    result = vkAllocateCommandBuffers(
        gRuntime.device,
        &commandBufferInfo,
        &gRuntime.commandBuffer
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan command-buffer allocation failed.", result);
        return false;
    }

    VkDescriptorSetLayoutBinding storageBinding{};
    storageBinding.binding = 0;
    storageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    storageBinding.descriptorCount = 1;
    storageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo{};
    descriptorLayoutInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorLayoutInfo.bindingCount = 1;
    descriptorLayoutInfo.pBindings = &storageBinding;

    result = vkCreateDescriptorSetLayout(
        gRuntime.device,
        &descriptorLayoutInfo,
        nullptr,
        &gRuntime.descriptorSetLayout
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan descriptor-set layout creation failed.", result);
        return false;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = static_cast<uint32_t>(
        std::max(sizeof(AdamwPushConstants), sizeof(LinearPushConstants))
    );

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &gRuntime.descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    result = vkCreatePipelineLayout(
        gRuntime.device,
        &pipelineLayoutInfo,
        nullptr,
        &gRuntime.pipelineLayout
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan pipeline-layout creation failed.", result);
        return false;
    }

    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = kArgusAdamwShaderSpvSize;
    shaderInfo.pCode =
        reinterpret_cast<const uint32_t*>(kArgusAdamwShaderSpv);

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    result = vkCreateShaderModule(
        gRuntime.device,
        &shaderInfo,
        nullptr,
        &shaderModule
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan AdamW shader module creation failed.", result);
        return false;
    }

    VkPipelineShaderStageCreateInfo shaderStage{};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = shaderModule;
    shaderStage.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStage;
    pipelineInfo.layout = gRuntime.pipelineLayout;

    result = vkCreateComputePipelines(
        gRuntime.device,
        VK_NULL_HANDLE,
        1,
        &pipelineInfo,
        nullptr,
        &gRuntime.pipeline
    );

    vkDestroyShaderModule(gRuntime.device, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        logError("Vulkan AdamW compute-pipeline creation failed.", result);
        return false;
    }

    VkShaderModuleCreateInfo linearShaderInfo{};
    linearShaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    linearShaderInfo.codeSize = kArgusLinearShaderSpvSize;
    linearShaderInfo.pCode =
        reinterpret_cast<const uint32_t*>(kArgusLinearShaderSpv);

    VkShaderModule linearShaderModule = VK_NULL_HANDLE;
    result = vkCreateShaderModule(
        gRuntime.device,
        &linearShaderInfo,
        nullptr,
        &linearShaderModule
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan linear shader module creation failed.", result);
        return false;
    }

    VkPipelineShaderStageCreateInfo linearShaderStage{};
    linearShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    linearShaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    linearShaderStage.module = linearShaderModule;
    linearShaderStage.pName = "main";

    VkComputePipelineCreateInfo linearPipelineInfo{};
    linearPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    linearPipelineInfo.stage = linearShaderStage;
    linearPipelineInfo.layout = gRuntime.pipelineLayout;

    result = vkCreateComputePipelines(
        gRuntime.device,
        VK_NULL_HANDLE,
        1,
        &linearPipelineInfo,
        nullptr,
        &gRuntime.linearPipeline
    );

    vkDestroyShaderModule(gRuntime.device, linearShaderModule, nullptr);

    if (result != VK_SUCCESS) {
        logError("Vulkan linear compute-pipeline creation failed.", result);
        return false;
    }

    if (gRuntime.supportsFp16Compute) {
        VkShaderModuleCreateInfo fp16ShaderInfo{};
        fp16ShaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        fp16ShaderInfo.codeSize = kArgusLinearFp16ShaderSpvSize;
        fp16ShaderInfo.pCode =
            reinterpret_cast<const uint32_t*>(kArgusLinearFp16ShaderSpv);

        VkShaderModule fp16ShaderModule = VK_NULL_HANDLE;
        VkResult fp16Result = vkCreateShaderModule(
            gRuntime.device,
            &fp16ShaderInfo,
            nullptr,
            &fp16ShaderModule
        );
        if (fp16Result == VK_SUCCESS) {
            VkPipelineShaderStageCreateInfo fp16ShaderStage{};
            fp16ShaderStage.sType =
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            fp16ShaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            fp16ShaderStage.module = fp16ShaderModule;
            fp16ShaderStage.pName = "main";

            VkComputePipelineCreateInfo fp16PipelineInfo{};
            fp16PipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            fp16PipelineInfo.stage = fp16ShaderStage;
            fp16PipelineInfo.layout = gRuntime.pipelineLayout;

            fp16Result = vkCreateComputePipelines(
                gRuntime.device,
                VK_NULL_HANDLE,
                1,
                &fp16PipelineInfo,
                nullptr,
                &gRuntime.fp16LinearPipeline
            );
            vkDestroyShaderModule(gRuntime.device, fp16ShaderModule, nullptr);
        }

        if (fp16Result != VK_SUCCESS) {
            logError("Vulkan FP16 linear pipeline is unavailable; retaining FP32 kernels.", fp16Result);
            gRuntime.supportsFp16Compute = false;
        }
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo descriptorPoolInfo{};
    descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolInfo.maxSets = 1;
    descriptorPoolInfo.poolSizeCount = 1;
    descriptorPoolInfo.pPoolSizes = &poolSize;

    result = vkCreateDescriptorPool(
        gRuntime.device,
        &descriptorPoolInfo,
        nullptr,
        &gRuntime.descriptorPool
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan descriptor-pool creation failed.", result);
        return false;
    }

    VkDescriptorSetAllocateInfo descriptorSetInfo{};
    descriptorSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetInfo.descriptorPool = gRuntime.descriptorPool;
    descriptorSetInfo.descriptorSetCount = 1;
    descriptorSetInfo.pSetLayouts = &gRuntime.descriptorSetLayout;

    result = vkAllocateDescriptorSets(
        gRuntime.device,
        &descriptorSetInfo,
        &gRuntime.descriptorSet
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan descriptor-set allocation failed.", result);
        return false;
    }

    __android_log_print(
        ANDROID_LOG_INFO,
        kLogTag,
        "Initialized Vulkan optimizer on %s (API %u.%u.%u).",
        gRuntime.physicalDeviceProperties.deviceName,
        VK_VERSION_MAJOR(gRuntime.physicalDeviceProperties.apiVersion),
        VK_VERSION_MINOR(gRuntime.physicalDeviceProperties.apiVersion),
        VK_VERSION_PATCH(gRuntime.physicalDeviceProperties.apiVersion)
    );

    return true;
}

uint16_t floatToHalf(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    const uint32_t absoluteBits = bits & 0x7FFFFFFFu;

    if (absoluteBits >= 0x7F800000u) {
        const uint32_t payload = absoluteBits == 0x7F800000u ? 0u : 0x0200u;
        return static_cast<uint16_t>(sign | 0x7C00u | payload);
    }

    const int32_t exponent =
        static_cast<int32_t>((bits >> 23u) & 0xFFu) - 127 + 15;
    const uint32_t mantissa = bits & 0x007FFFFFu;

    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }

        const uint32_t significand = mantissa | 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint32_t rounded = significand >> shift;
        const uint32_t remainderMask = (1u << shift) - 1u;
        const uint32_t remainder = significand & remainderMask;
        const uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway ||
            (remainder == halfway && (rounded & 1u) != 0u)) {
            ++rounded;
        }
        return static_cast<uint16_t>(sign | rounded);
    }

    uint32_t halfMantissa = mantissa >> 13u;
    const uint32_t discarded = mantissa & 0x1FFFu;
    if (discarded > 0x1000u ||
        (discarded == 0x1000u && (halfMantissa & 1u) != 0u)) {
        ++halfMantissa;
    }

    int32_t halfExponent = exponent;
    if (halfMantissa == 0x0400u) {
        halfMantissa = 0u;
        ++halfExponent;
        if (halfExponent >= 31) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
    }

    return static_cast<uint16_t>(
        sign |
        (static_cast<uint32_t>(halfExponent) << 10u) |
        halfMantissa
    );
}

void releaseGroupReferences(JNIEnv* env, std::vector<GroupArrays>& groups) {
    for (GroupArrays& group : groups) {
        if (group.parameters != nullptr) {
            env->DeleteLocalRef(group.parameters);
        }
        if (group.gradients != nullptr) {
            env->DeleteLocalRef(group.gradients);
        }
        if (group.firstMoments != nullptr) {
            env->DeleteLocalRef(group.firstMoments);
        }
        if (group.secondMoments != nullptr) {
            env->DeleteLocalRef(group.secondMoments);
        }
    }
}

} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeInitialize(
    JNIEnv*,
    jobject
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    destroyRuntime();

    if (initializeRuntime()) {
        return JNI_TRUE;
    }

    destroyRuntime();
    return JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeSupportsFp16(
    JNIEnv*,
    jobject
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    return gRuntime.device != VK_NULL_HANDLE &&
        gRuntime.supportsFp16Compute &&
        gRuntime.fp16LinearPipeline != VK_NULL_HANDLE
        ? JNI_TRUE
        : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeApplyAdamWUpdate(
    JNIEnv* env,
    jobject,
    jobjectArray parameterGroups,
    jobjectArray gradientGroups,
    jobjectArray firstMomentGroups,
    jobjectArray secondMomentGroups,
    jfloatArray weightDecayByGroup,
    jfloat learningRate,
    jfloat beta1,
    jfloat beta2,
    jfloat biasCorrection1,
    jfloat biasCorrection2,
    jfloat epsilon,
    jfloat gradientClipFactor
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);

    if (gRuntime.device == VK_NULL_HANDLE ||
        parameterGroups == nullptr ||
        gradientGroups == nullptr ||
        firstMomentGroups == nullptr ||
        secondMomentGroups == nullptr ||
        weightDecayByGroup == nullptr) {
        return JNI_FALSE;
    }

    const jsize groupCount = env->GetArrayLength(parameterGroups);
    if (env->ExceptionCheck() || groupCount <= 0 ||
        env->GetArrayLength(gradientGroups) != groupCount ||
        env->GetArrayLength(firstMomentGroups) != groupCount ||
        env->GetArrayLength(secondMomentGroups) != groupCount ||
        env->GetArrayLength(weightDecayByGroup) != groupCount) {
        return JNI_FALSE;
    }

    std::vector<GroupArrays> groups;
    std::vector<AdamwSegment> segments;
    try {
        groups.resize(static_cast<size_t>(groupCount));
        segments.reserve(static_cast<size_t>(groupCount));
    } catch (const std::bad_alloc&) {
        return JNI_FALSE;
    }

    size_t totalElements = 0;
    for (jsize index = 0; index < groupCount; ++index) {
        GroupArrays& group = groups[static_cast<size_t>(index)];

        group.parameters = static_cast<jfloatArray>(
            env->GetObjectArrayElement(parameterGroups, index)
        );
        group.gradients = static_cast<jfloatArray>(
            env->GetObjectArrayElement(gradientGroups, index)
        );
        group.firstMoments = static_cast<jfloatArray>(
            env->GetObjectArrayElement(firstMomentGroups, index)
        );
        group.secondMoments = static_cast<jfloatArray>(
            env->GetObjectArrayElement(secondMomentGroups, index)
        );

        if (env->ExceptionCheck() ||
            group.parameters == nullptr ||
            group.gradients == nullptr ||
            group.firstMoments == nullptr ||
            group.secondMoments == nullptr) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }

        const jsize parameterCount = env->GetArrayLength(group.parameters);
        const jsize gradientCount = env->GetArrayLength(group.gradients);
        const jsize firstMomentCount = env->GetArrayLength(group.firstMoments);
        const jsize secondMomentCount = env->GetArrayLength(group.secondMoments);

        if (env->ExceptionCheck() ||
            parameterCount <= 0 ||
            gradientCount != parameterCount ||
            firstMomentCount != parameterCount ||
            secondMomentCount != parameterCount) {
            logError("Vulkan AdamW parameter, gradient, and moment group sizes must match.");
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }
        group.elementCount = parameterCount;

        env->GetFloatArrayRegion(weightDecayByGroup, index, 1, &group.weightDecay);
        if (env->ExceptionCheck() ||
            !std::isfinite(group.weightDecay) ||
            group.weightDecay < 0.0f) {
            logError("Vulkan AdamW weight decay must be finite and non-negative.");
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }
        if (group.weightDecay == 0.0f) {
            group.weightDecay = 0.0f;
        }

        const size_t count = static_cast<size_t>(group.elementCount);
        if (totalElements > std::numeric_limits<size_t>::max() - count) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }
        totalElements += count;
    }

    if (totalElements == 0 ||
        totalElements > std::numeric_limits<uint32_t>::max() ||
        totalElements > std::numeric_limits<size_t>::max() / (4u * sizeof(float))) {
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }

    // Sort tensors by decay rate so all tensors sharing the same scalar
    // weight decay occupy one contiguous range. This removes the per-element
    // decay array while preserving the original Java array references.
    std::sort(
        groups.begin(),
        groups.end(),
        [](const GroupArrays& left, const GroupArrays& right) {
            return left.weightDecay < right.weightDecay;
        }
    );

    size_t packedOffset = 0;
    for (GroupArrays& group : groups) {
        group.packedOffset = packedOffset;
        const uint32_t groupElementCount =
            static_cast<uint32_t>(group.elementCount);

        if (segments.empty() ||
            segments.back().weightDecay != group.weightDecay) {
            segments.push_back(AdamwSegment{
                static_cast<uint32_t>(packedOffset),
                groupElementCount,
                group.weightDecay
            });
        } else {
            AdamwSegment& segment = segments.back();
            if (groupElementCount >
                std::numeric_limits<uint32_t>::max() - segment.elementCount) {
                releaseGroupReferences(env, groups);
                return JNI_FALSE;
            }
            segment.elementCount += groupElementCount;
        }

        packedOffset += static_cast<size_t>(groupElementCount);
    }

    if (packedOffset != totalElements || segments.empty()) {
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }

    for (const AdamwSegment& segment : segments) {
        if (calculateDispatchWorkGroupCount(segment.elementCount) == 0u) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }
    }

    const size_t packedFloatCount = totalElements * 4u;
    const size_t byteCount = packedFloatCount * sizeof(float);
    if (byteCount > static_cast<size_t>(
            gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange
        )) {
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }

    if (!ensureStorageBuffer(static_cast<VkDeviceSize>(byteCount))) {
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }

    // One mapped allocation stages four tensor arrays. Weight decay is now a
    // scalar per contiguous segment, not a fifth full-size tensor.
    float* parameterValues = static_cast<float*>(gRuntime.mappedStorage);
    float* gradientValues = parameterValues + totalElements;
    float* firstMomentValues = gradientValues + totalElements;
    float* secondMomentValues = firstMomentValues + totalElements;

    for (const GroupArrays& group : groups) {
        const size_t offset = group.packedOffset;
        const jsize count = group.elementCount;

        env->GetFloatArrayRegion(
            group.parameters,
            0,
            count,
            parameterValues + offset
        );
        if (env->ExceptionCheck()) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }

        env->GetFloatArrayRegion(
            group.gradients,
            0,
            count,
            gradientValues + offset
        );
        if (env->ExceptionCheck()) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }

        env->GetFloatArrayRegion(
            group.firstMoments,
            0,
            count,
            firstMomentValues + offset
        );
        if (env->ExceptionCheck()) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }

        env->GetFloatArrayRegion(
            group.secondMoments,
            0,
            count,
            secondMomentValues + offset
        );
        if (env->ExceptionCheck()) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }
    }

    if (!gRuntime.hostCoherent) {
        VkMappedMemoryRange flushRange{};
        flushRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        flushRange.memory = gRuntime.storageMemory;
        flushRange.offset = 0;
        flushRange.size = VK_WHOLE_SIZE;

        const VkResult flushResult = vkFlushMappedMemoryRanges(
            gRuntime.device,
            1,
            &flushRange
        );
        if (flushResult != VK_SUCCESS) {
            logError("Failed to flush Vulkan host-visible memory.", flushResult);
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }
    }

    VkResult result = vkResetCommandBuffer(gRuntime.commandBuffer, 0);
    if (result != VK_SUCCESS) {
        logError("Failed to reset the Vulkan command buffer.", result);
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    result = vkBeginCommandBuffer(gRuntime.commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        logError("Failed to begin the Vulkan command buffer.", result);
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }

    VkMemoryBarrier hostToComputeBarrier{};
    hostToComputeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hostToComputeBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    hostToComputeBarrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        gRuntime.commandBuffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &hostToComputeBarrier,
        0,
        nullptr,
        0,
        nullptr
    );

    vkCmdBindPipeline(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.pipeline
    );
    vkCmdBindDescriptorSets(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.pipelineLayout,
        0,
        1,
        &gRuntime.descriptorSet,
        0,
        nullptr
    );

    for (const AdamwSegment& segment : segments) {
        const AdamwPushConstants pushConstants{
            static_cast<uint32_t>(totalElements),
            segment.elementOffset,
            segment.elementCount,
            learningRate,
            beta1,
            beta2,
            biasCorrection1,
            biasCorrection2,
            epsilon,
            gradientClipFactor,
            segment.weightDecay
        };

        vkCmdPushConstants(
            gRuntime.commandBuffer,
            gRuntime.pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(AdamwPushConstants),
            &pushConstants
        );

        const uint32_t workGroupCount =
            calculateDispatchWorkGroupCount(segment.elementCount);
        vkCmdDispatch(
            gRuntime.commandBuffer,
            workGroupCount,
            1,
            1
        );
    }

    VkMemoryBarrier computeToHostBarrier{};
    computeToHostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    computeToHostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    computeToHostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        gRuntime.commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &computeToHostBarrier,
        0,
        nullptr,
        0,
        nullptr
    );

    result = vkEndCommandBuffer(gRuntime.commandBuffer);
    if (result != VK_SUCCESS) {
        logError("Failed to finish the Vulkan command buffer.", result);
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &gRuntime.commandBuffer;

    result = vkQueueSubmit(
        gRuntime.computeQueue,
        1,
        &submitInfo,
        VK_NULL_HANDLE
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan optimizer queue submission failed.", result);
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }

    result = vkQueueWaitIdle(gRuntime.computeQueue);
    if (result != VK_SUCCESS) {
        logError("Vulkan optimizer queue did not complete.", result);
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }

    if (!gRuntime.hostCoherent) {
        VkMappedMemoryRange invalidateRange{};
        invalidateRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        invalidateRange.memory = gRuntime.storageMemory;
        invalidateRange.offset = 0;
        invalidateRange.size = VK_WHOLE_SIZE;

        result = vkInvalidateMappedMemoryRanges(
            gRuntime.device,
            1,
            &invalidateRange
        );
        if (result != VK_SUCCESS) {
            logError("Failed to invalidate Vulkan host-visible memory.", result);
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }
    }

    const float* outputParameters =
        static_cast<const float*>(gRuntime.mappedStorage);
    const float* outputFirstMoments = outputParameters + totalElements * 2u;
    const float* outputSecondMoments = outputParameters + totalElements * 3u;

    // Validate every result before changing any Java-side model or optimizer tensor.
    for (size_t index = 0; index < totalElements; ++index) {
        if (!std::isfinite(outputParameters[index]) ||
            !std::isfinite(outputFirstMoments[index]) ||
            !std::isfinite(outputSecondMoments[index])) {
            logError("Vulkan AdamW produced a non-finite parameter or optimizer-state value.");
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }
    }

    for (const GroupArrays& group : groups) {
        const size_t offset = group.packedOffset;
        const jsize count = group.elementCount;

        env->SetFloatArrayRegion(
            group.parameters,
            0,
            count,
            outputParameters + offset
        );
        env->SetFloatArrayRegion(
            group.firstMoments,
            0,
            count,
            outputFirstMoments + offset
        );
        env->SetFloatArrayRegion(
            group.secondMoments,
            0,
            count,
            outputSecondMoments + offset
        );

        if (env->ExceptionCheck()) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }
    }

    releaseGroupReferences(env, groups);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeApplyLinear(
    JNIEnv* env,
    jobject,
    jfloatArray inputArray,
    jfloatArray weightArray,
    jfloatArray outputArray,
    jint rowCount,
    jint batchCount,
    jint rowsPerBatch,
    jint inputColumns,
    jint outputColumns,
    jboolean transposeWeights
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);

    if (gRuntime.device == VK_NULL_HANDLE ||
        gRuntime.linearPipeline == VK_NULL_HANDLE ||
        inputArray == nullptr ||
        weightArray == nullptr ||
        outputArray == nullptr ||
        rowCount <= 0 ||
        batchCount <= 0 ||
        rowsPerBatch <= 0 ||
        static_cast<int64_t>(batchCount) * static_cast<int64_t>(rowsPerBatch) != rowCount ||
        inputColumns <= 0 ||
        outputColumns <= 0) {
        return JNI_FALSE;
    }

    const uint64_t inputElements =
        static_cast<uint64_t>(rowCount) * static_cast<uint64_t>(inputColumns);
    // Bound the intermediate product before multiplying by outputColumns.
    // The JNI array length must fit jsize, so rejecting earlier also prevents
    // uint64_t wraparound in the three-dimensional weight count.
    const uint64_t maximumArrayElements =
        static_cast<uint64_t>(std::numeric_limits<jsize>::max());
    const uint64_t batchInputWidth = static_cast<uint64_t>(batchCount);
    const uint64_t inputColumnCount = static_cast<uint64_t>(inputColumns);
    const uint64_t outputColumnCount = static_cast<uint64_t>(outputColumns);
    if (batchInputWidth > maximumArrayElements / inputColumnCount) {
        return JNI_FALSE;
    }
    const uint64_t batchInputElements = batchInputWidth * inputColumnCount;
    if (batchInputElements > maximumArrayElements / outputColumnCount) {
        return JNI_FALSE;
    }
    const uint64_t weightElements = batchInputElements * outputColumnCount;
    const uint64_t outputElements =
        static_cast<uint64_t>(rowCount) * static_cast<uint64_t>(outputColumns);
    const uint64_t totalElements = inputElements + weightElements + outputElements;

    if (inputElements > static_cast<uint64_t>(std::numeric_limits<jsize>::max()) ||
        weightElements > static_cast<uint64_t>(std::numeric_limits<jsize>::max()) ||
        outputElements > static_cast<uint64_t>(std::numeric_limits<jsize>::max()) ||
        totalElements > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        totalElements > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float))) {
        return JNI_FALSE;
    }

    if (env->GetArrayLength(inputArray) != static_cast<jsize>(inputElements) ||
        env->GetArrayLength(weightArray) < static_cast<jsize>(weightElements) ||
        env->GetArrayLength(outputArray) != static_cast<jsize>(outputElements)) {
        return JNI_FALSE;
    }

    const size_t inputOffset = 0u;
    const size_t weightOffset = static_cast<size_t>(inputElements);
    const size_t outputOffset = static_cast<size_t>(inputElements + weightElements);
    const size_t totalByteCount = static_cast<size_t>(totalElements) * sizeof(float);

    if (totalByteCount > static_cast<size_t>(
            gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange
        )) {
        return JNI_FALSE;
    }

    if (!ensureStorageBuffer(static_cast<VkDeviceSize>(totalByteCount))) {
        logError("Vulkan linear storage-buffer allocation failed.");
        return JNI_FALSE;
    }

    // Copy JNI arrays directly into the mapped Vulkan staging allocation.
    // This avoids a temporary tensor vector and a second full-buffer memcpy.
    float* packedValues = static_cast<float*>(gRuntime.mappedStorage);
    env->GetFloatArrayRegion(
        inputArray,
        0,
        static_cast<jsize>(inputElements),
        packedValues + inputOffset
    );
    if (env->ExceptionCheck()) {
        return JNI_FALSE;
    }

    env->GetFloatArrayRegion(
        weightArray,
        0,
        static_cast<jsize>(weightElements),
        packedValues + weightOffset
    );
    if (env->ExceptionCheck()) {
        return JNI_FALSE;
    }

    if (!gRuntime.hostCoherent) {
        VkMappedMemoryRange flushRange{};
        flushRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        flushRange.memory = gRuntime.storageMemory;
        flushRange.offset = 0;
        flushRange.size = VK_WHOLE_SIZE;

        const VkResult flushResult = vkFlushMappedMemoryRanges(
            gRuntime.device,
            1,
            &flushRange
        );
        if (flushResult != VK_SUCCESS) {
            logError("Failed to flush Vulkan linear input memory.", flushResult);
            return JNI_FALSE;
        }
    }

    const uint32_t workGroupCount =
        calculateDispatchWorkGroupCount(outputElements);
    if (workGroupCount == 0u) {
        logError("Vulkan linear dispatch exceeds the device work-group limit.");
        return JNI_FALSE;
    }

    VkResult result = vkResetCommandBuffer(gRuntime.commandBuffer, 0);
    if (result != VK_SUCCESS) {
        logError("Failed to reset the Vulkan linear command buffer.", result);
        return JNI_FALSE;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    result = vkBeginCommandBuffer(gRuntime.commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        logError("Failed to begin the Vulkan linear command buffer.", result);
        return JNI_FALSE;
    }

    VkMemoryBarrier hostToComputeBarrier{};
    hostToComputeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hostToComputeBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    hostToComputeBarrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        gRuntime.commandBuffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &hostToComputeBarrier,
        0,
        nullptr,
        0,
        nullptr
    );

    vkCmdBindPipeline(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.linearPipeline
    );
    vkCmdBindDescriptorSets(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.pipelineLayout,
        0,
        1,
        &gRuntime.descriptorSet,
        0,
        nullptr
    );

    const LinearPushConstants pushConstants{
        static_cast<uint32_t>(rowCount),
        static_cast<uint32_t>(batchCount),
        static_cast<uint32_t>(rowsPerBatch),
        static_cast<uint32_t>(inputColumns),
        static_cast<uint32_t>(outputColumns),
        static_cast<uint32_t>(inputOffset),
        static_cast<uint32_t>(weightOffset),
        static_cast<uint32_t>(outputOffset),
        transposeWeights == JNI_TRUE ? 1u : 0u
    };

    vkCmdPushConstants(
        gRuntime.commandBuffer,
        gRuntime.pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(LinearPushConstants),
        &pushConstants
    );

    vkCmdDispatch(
        gRuntime.commandBuffer,
        static_cast<uint32_t>(workGroupCount),
        1,
        1
    );

    VkMemoryBarrier computeToHostBarrier{};
    computeToHostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    computeToHostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    computeToHostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        gRuntime.commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &computeToHostBarrier,
        0,
        nullptr,
        0,
        nullptr
    );

    result = vkEndCommandBuffer(gRuntime.commandBuffer);
    if (result != VK_SUCCESS) {
        logError("Failed to finish the Vulkan linear command buffer.", result);
        return JNI_FALSE;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &gRuntime.commandBuffer;

    result = vkQueueSubmit(gRuntime.computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        logError("Vulkan linear queue submission failed.", result);
        return JNI_FALSE;
    }

    result = vkQueueWaitIdle(gRuntime.computeQueue);
    if (result != VK_SUCCESS) {
        logError("Vulkan linear queue did not complete.", result);
        return JNI_FALSE;
    }

    if (!gRuntime.hostCoherent) {
        VkMappedMemoryRange invalidateRange{};
        invalidateRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        invalidateRange.memory = gRuntime.storageMemory;
        invalidateRange.offset = 0;
        invalidateRange.size = VK_WHOLE_SIZE;

        result = vkInvalidateMappedMemoryRanges(
            gRuntime.device,
            1,
            &invalidateRange
        );
        if (result != VK_SUCCESS) {
            logError("Failed to invalidate Vulkan linear output memory.", result);
            return JNI_FALSE;
        }
    }

    const float* mappedValues = static_cast<const float*>(gRuntime.mappedStorage);
    env->SetFloatArrayRegion(
        outputArray,
        0,
        static_cast<jsize>(outputElements),
        mappedValues + outputOffset
    );
    if (env->ExceptionCheck()) {
        return JNI_FALSE;
    }

    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeApplyFp16Linear(
    JNIEnv* env,
    jobject,
    jfloatArray inputArray,
    jshortArray weightArray,
    jfloatArray outputArray,
    jint rowCount,
    jint batchCount,
    jint rowsPerBatch,
    jint inputColumns,
    jint outputColumns,
    jboolean transposeWeights
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);

    if (gRuntime.device == VK_NULL_HANDLE ||
        !gRuntime.supportsFp16Compute ||
        gRuntime.fp16LinearPipeline == VK_NULL_HANDLE ||
        inputArray == nullptr ||
        weightArray == nullptr ||
        outputArray == nullptr ||
        rowCount <= 0 ||
        batchCount <= 0 ||
        rowsPerBatch <= 0 ||
        static_cast<int64_t>(batchCount) * static_cast<int64_t>(rowsPerBatch) != rowCount ||
        inputColumns <= 0 ||
        outputColumns <= 0) {
        return JNI_FALSE;
    }

    const uint64_t inputElements =
        static_cast<uint64_t>(rowCount) * static_cast<uint64_t>(inputColumns);
    // Bound the intermediate product before multiplying by outputColumns.
    // The JNI array length must fit jsize, so rejecting earlier also prevents
    // uint64_t wraparound in the three-dimensional weight count.
    const uint64_t maximumArrayElements =
        static_cast<uint64_t>(std::numeric_limits<jsize>::max());
    const uint64_t batchInputWidth = static_cast<uint64_t>(batchCount);
    const uint64_t inputColumnCount = static_cast<uint64_t>(inputColumns);
    const uint64_t outputColumnCount = static_cast<uint64_t>(outputColumns);
    if (batchInputWidth > maximumArrayElements / inputColumnCount) {
        return JNI_FALSE;
    }
    const uint64_t batchInputElements = batchInputWidth * inputColumnCount;
    if (batchInputElements > maximumArrayElements / outputColumnCount) {
        return JNI_FALSE;
    }
    const uint64_t weightElements = batchInputElements * outputColumnCount;
    const uint64_t outputElements =
        static_cast<uint64_t>(rowCount) * static_cast<uint64_t>(outputColumns);

    if (inputElements > static_cast<uint64_t>(std::numeric_limits<jsize>::max()) ||
        weightElements > static_cast<uint64_t>(std::numeric_limits<jsize>::max()) ||
        outputElements > static_cast<uint64_t>(std::numeric_limits<jsize>::max()) ||
        inputElements > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        weightElements > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        outputElements > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return JNI_FALSE;
    }

    if (env->GetArrayLength(inputArray) != static_cast<jsize>(inputElements) ||
        env->GetArrayLength(weightArray) != static_cast<jsize>(weightElements) ||
        env->GetArrayLength(outputArray) != static_cast<jsize>(outputElements)) {
        return JNI_FALSE;
    }

    const uint64_t weightOffsetHalf = (inputElements + 1u) & ~1ull;
    const uint64_t outputOffsetWords =
        (weightOffsetHalf + weightElements + 1u) / 2u;
    const uint64_t totalWords = outputOffsetWords + outputElements;
    if (weightOffsetHalf > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        outputOffsetWords > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        totalWords > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(uint32_t))) {
        return JNI_FALSE;
    }

    const size_t totalByteCount = static_cast<size_t>(totalWords) * sizeof(uint32_t);
    if (totalByteCount > static_cast<size_t>(
            gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange
        )) {
        return JNI_FALSE;
    }

    if (!ensureStorageBuffer(static_cast<VkDeviceSize>(totalByteCount))) {
        logError("Vulkan FP16 linear storage-buffer allocation failed.");
        return JNI_FALSE;
    }

    std::vector<jfloat> inputValues;
    try {
        inputValues.resize(static_cast<size_t>(inputElements));
    } catch (const std::bad_alloc&) {
        logError("Not enough host memory to prepare Vulkan FP16 linear input.");
        return JNI_FALSE;
    }

    env->GetFloatArrayRegion(
        inputArray,
        0,
        static_cast<jsize>(inputElements),
        inputValues.data()
    );
    if (env->ExceptionCheck()) {
        return JNI_FALSE;
    }

    auto* packedWords = static_cast<uint32_t*>(gRuntime.mappedStorage);
    for (size_t index = 0; index < inputValues.size(); index += 2u) {
        const float firstValue = inputValues[index];
        if (!std::isfinite(firstValue)) {
            return JNI_FALSE;
        }

        const uint32_t lowHalf = floatToHalf(firstValue);
        uint32_t highHalf = 0u;
        if (index + 1u < inputValues.size()) {
            const float secondValue = inputValues[index + 1u];
            if (!std::isfinite(secondValue)) {
                return JNI_FALSE;
            }
            highHalf = static_cast<uint32_t>(floatToHalf(secondValue)) << 16u;
        }
        packedWords[index >> 1u] = lowHalf | highHalf;
    }

    // Weight arrays already contain IEEE-754 binary16 payloads. Copy them
    // directly into mapped storage instead of allocating and repacking a
    // second native vector.
    auto* packedWeights = reinterpret_cast<jshort*>(packedWords) + weightOffsetHalf;
    env->GetShortArrayRegion(
        weightArray,
        0,
        static_cast<jsize>(weightElements),
        packedWeights
    );
    if (env->ExceptionCheck()) {
        return JNI_FALSE;
    }

    if (!gRuntime.hostCoherent) {
        VkMappedMemoryRange flushRange{};
        flushRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        flushRange.memory = gRuntime.storageMemory;
        flushRange.offset = 0;
        flushRange.size = VK_WHOLE_SIZE;
        const VkResult flushResult = vkFlushMappedMemoryRanges(
            gRuntime.device,
            1,
            &flushRange
        );
        if (flushResult != VK_SUCCESS) {
            logError("Failed to flush Vulkan FP16 linear input memory.", flushResult);
            return JNI_FALSE;
        }
    }

    // The FP16 shader uses a grid-stride loop, so clamp the physical
    // dispatch to the device X limit and let each invocation cover its
    // additional output elements through the shader stride.
    const uint32_t workGroupCount =
        calculateDispatchWorkGroupCount(outputElements);
    if (workGroupCount == 0u) {
        logError("Vulkan FP16 linear dispatch has no available compute work-groups.");
        return JNI_FALSE;
    }

    VkResult result = vkResetCommandBuffer(gRuntime.commandBuffer, 0);
    if (result != VK_SUCCESS) {
        logError("Failed to reset the Vulkan FP16 linear command buffer.", result);
        return JNI_FALSE;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(gRuntime.commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        logError("Failed to begin the Vulkan FP16 linear command buffer.", result);
        return JNI_FALSE;
    }

    VkMemoryBarrier hostToComputeBarrier{};
    hostToComputeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hostToComputeBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    hostToComputeBarrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        gRuntime.commandBuffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &hostToComputeBarrier,
        0,
        nullptr,
        0,
        nullptr
    );

    vkCmdBindPipeline(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.fp16LinearPipeline
    );
    vkCmdBindDescriptorSets(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.pipelineLayout,
        0,
        1,
        &gRuntime.descriptorSet,
        0,
        nullptr
    );

    const LinearPushConstants pushConstants{
        static_cast<uint32_t>(rowCount),
        static_cast<uint32_t>(batchCount),
        static_cast<uint32_t>(rowsPerBatch),
        static_cast<uint32_t>(inputColumns),
        static_cast<uint32_t>(outputColumns),
        0u,
        static_cast<uint32_t>(weightOffsetHalf),
        static_cast<uint32_t>(outputOffsetWords),
        transposeWeights == JNI_TRUE ? 1u : 0u
    };
    vkCmdPushConstants(
        gRuntime.commandBuffer,
        gRuntime.pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(LinearPushConstants),
        &pushConstants
    );
    vkCmdDispatch(
        gRuntime.commandBuffer,
        static_cast<uint32_t>(workGroupCount),
        1,
        1
    );

    VkMemoryBarrier computeToHostBarrier{};
    computeToHostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    computeToHostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    computeToHostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(
        gRuntime.commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &computeToHostBarrier,
        0,
        nullptr,
        0,
        nullptr
    );

    result = vkEndCommandBuffer(gRuntime.commandBuffer);
    if (result != VK_SUCCESS) {
        logError("Failed to finish the Vulkan FP16 linear command buffer.", result);
        return JNI_FALSE;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &gRuntime.commandBuffer;
    result = vkQueueSubmit(gRuntime.computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        logError("Vulkan FP16 linear queue submission failed.", result);
        return JNI_FALSE;
    }

    result = vkQueueWaitIdle(gRuntime.computeQueue);
    if (result != VK_SUCCESS) {
        logError("Vulkan FP16 linear queue did not complete.", result);
        return JNI_FALSE;
    }

    if (!gRuntime.hostCoherent) {
        VkMappedMemoryRange invalidateRange{};
        invalidateRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        invalidateRange.memory = gRuntime.storageMemory;
        invalidateRange.offset = 0;
        invalidateRange.size = VK_WHOLE_SIZE;
        result = vkInvalidateMappedMemoryRanges(
            gRuntime.device,
            1,
            &invalidateRange
        );
        if (result != VK_SUCCESS) {
            logError("Failed to invalidate Vulkan FP16 linear output memory.", result);
            return JNI_FALSE;
        }
    }

    const auto* mappedWords = static_cast<const uint32_t*>(gRuntime.mappedStorage);
    const auto* outputValues =
        reinterpret_cast<const jfloat*>(mappedWords + outputOffsetWords);
    env->SetFloatArrayRegion(
        outputArray,
        0,
        static_cast<jsize>(outputElements),
        outputValues
    );
    if (env->ExceptionCheck()) {
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeDestroy(
    JNIEnv*,
    jobject
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    destroyRuntime();
}
