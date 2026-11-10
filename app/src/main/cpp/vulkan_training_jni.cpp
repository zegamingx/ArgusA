#include <jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>

#include "adamw_shader_spv.h"

#include <algorithm>
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
    uint32_t elementCount;
    float learningRate;
    float beta1;
    float beta2;
    float biasCorrection1;
    float biasCorrection2;
    float epsilon;
    float gradientClipFactor;
};

static_assert(sizeof(AdamwPushConstants) == 28);

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
    applicationInfo.apiVersion = VK_API_VERSION_1_0;

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

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.pEnabledFeatures = &deviceFeatures;

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
    pushConstantRange.size = sizeof(AdamwPushConstants);

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
    if (groupCount <= 0 ||
        env->GetArrayLength(gradientGroups) != groupCount ||
        env->GetArrayLength(firstMomentGroups) != groupCount ||
        env->GetArrayLength(secondMomentGroups) != groupCount ||
        env->GetArrayLength(weightDecayByGroup) != groupCount) {
        return JNI_FALSE;
    }

    std::vector<GroupArrays> groups;
    try {
        groups.resize(static_cast<size_t>(groupCount));
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

        group.elementCount = std::min(
            std::min(parameterCount, gradientCount),
            std::min(firstMomentCount, secondMomentCount)
        );

        if (group.elementCount <= 0) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }

        group.packedOffset = totalElements;
        const size_t count = static_cast<size_t>(group.elementCount);
        if (totalElements > std::numeric_limits<size_t>::max() - count) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }
        totalElements += count;
    }

    if (totalElements == 0 ||
        totalElements > std::numeric_limits<uint32_t>::max() ||
        totalElements > std::numeric_limits<size_t>::max() / (5u * sizeof(float))) {
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }

    const size_t packedFloatCount = totalElements * 5u;
    std::vector<float> packed;
    try {
        packed.resize(packedFloatCount);
    } catch (const std::bad_alloc&) {
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }

    float* parameterValues = packed.data();
    float* gradientValues = parameterValues + totalElements;
    float* firstMomentValues = gradientValues + totalElements;
    float* secondMomentValues = firstMomentValues + totalElements;
    float* decayValues = secondMomentValues + totalElements;

    for (jsize index = 0; index < groupCount; ++index) {
        GroupArrays& group = groups[static_cast<size_t>(index)];
        const size_t offset = group.packedOffset;
        const jsize count = group.elementCount;

        env->GetFloatArrayRegion(
            group.parameters,
            0,
            count,
            parameterValues + offset
        );
        env->GetFloatArrayRegion(
            group.gradients,
            0,
            count,
            gradientValues + offset
        );
        env->GetFloatArrayRegion(
            group.firstMoments,
            0,
            count,
            firstMomentValues + offset
        );
        env->GetFloatArrayRegion(
            group.secondMoments,
            0,
            count,
            secondMomentValues + offset
        );

        jfloat decay = 0.0f;
        env->GetFloatArrayRegion(weightDecayByGroup, index, 1, &decay);

        if (env->ExceptionCheck()) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }

        std::fill(
            decayValues + offset,
            decayValues + offset + static_cast<size_t>(count),
            decay
        );
    }

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

    std::memcpy(gRuntime.mappedStorage, packed.data(), byteCount);

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

    const uint64_t workGroupCount =
        (static_cast<uint64_t>(totalElements) + kLocalWorkGroupSize - 1u) /
        kLocalWorkGroupSize;

    if (workGroupCount == 0 ||
        workGroupCount > gRuntime.maxComputeWorkGroupCountX) {
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
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

    const AdamwPushConstants pushConstants{
        static_cast<uint32_t>(totalElements),
        learningRate,
        beta1,
        beta2,
        biasCorrection1,
        biasCorrection2,
        epsilon,
        gradientClipFactor
    };

    vkCmdPushConstants(
        gRuntime.commandBuffer,
        gRuntime.pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(AdamwPushConstants),
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

extern "C" JNIEXPORT void JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeDestroy(
    JNIEnv*,
    jobject
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    destroyRuntime();
}
