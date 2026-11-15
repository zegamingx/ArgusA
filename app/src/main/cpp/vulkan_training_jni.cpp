#include <jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>

#include "adamw_shader_spv.h"
#include "linear_shader_spv.h"
#include "linear_fp16_shader_spv.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
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

struct CooperativeMatrixCapability {
    uint32_t mSize = 0;
    uint32_t nSize = 0;
    uint32_t kSize = 0;
    uint32_t aType = 0;
    uint32_t bType = 0;
    uint32_t cType = 0;
    uint32_t resultType = 0;
    bool saturatingAccumulation = false;
    uint32_t scope = 0;
};

struct VulkanRuntime {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties physicalDeviceProperties{};
    VkPhysicalDeviceDriverProperties driverProperties{};
    bool hasDriverProperties = false;
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    uint32_t maxComputeWorkGroupCountX = 0;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence submissionFence = VK_NULL_HANDLE;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipeline linearPipeline = VK_NULL_HANDLE;
    VkPipeline fp16LinearPipeline = VK_NULL_HANDLE;

    uint32_t instanceApiVersion = VK_API_VERSION_1_0;
    bool supportsFp16Compute = false;
    bool supportsShaderInt8 = false;
    bool supportsStorageBuffer16BitAccess = false;

    uint32_t subgroupSize = 0;
    uint32_t subgroupSupportedStages = 0;
    uint32_t subgroupSupportedOperations = 0;
    bool subgroupQuadOperationsInAllStages = false;
    bool supportsSubgroupSizeControl = false;
    bool supportsComputeFullSubgroups = false;
    uint32_t minSubgroupSize = 0;
    uint32_t maxSubgroupSize = 0;
    uint32_t maxComputeWorkgroupSubgroups = 0;
    uint32_t requiredSubgroupSizeStages = 0;

    bool supportsShaderIntegerDotProduct = false;
    bool integerDotProduct8BitUnsignedAccelerated = false;
    bool integerDotProduct8BitSignedAccelerated = false;
    bool integerDotProduct8BitMixedSignednessAccelerated = false;
    bool integerDotProduct4x8BitPackedUnsignedAccelerated = false;
    bool integerDotProduct4x8BitPackedSignedAccelerated = false;
    bool integerDotProduct4x8BitPackedMixedSignednessAccelerated = false;

    bool hasCooperativeMatrixExtension = false;
    bool supportsCooperativeMatrix = false;
    bool supportsCooperativeMatrixRobustBufferAccess = false;
    uint32_t cooperativeMatrixSupportedStages = 0;
    std::vector<CooperativeMatrixCapability> cooperativeMatrixTypes;

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

bool hasDeviceExtension(
    const std::vector<VkExtensionProperties>& extensions,
    const char* extensionName
) {
    return std::any_of(
        extensions.begin(),
        extensions.end(),
        [extensionName](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, extensionName) == 0;
        }
    );
}

std::string escapeJsonString(const char* value) {
    std::string escaped;
    if (value == nullptr) {
        return escaped;
    }

    for (const unsigned char character : std::string(value)) {
        switch (character) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\b':
                escaped += "\\b";
                break;
            case '\f':
                escaped += "\\f";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (character < 0x20u) {
                    char buffer[7]{};
                    std::snprintf(
                        buffer,
                        sizeof(buffer),
                        "\\u%04X",
                        static_cast<unsigned int>(character)
                    );
                    escaped += buffer;
                } else {
                    escaped += static_cast<char>(character);
                }
                break;
        }
    }
    return escaped;
}

uint32_t calculateDispatchGroupCount(uint64_t requiredGroupCount) {
    const uint32_t deviceGroupLimit = gRuntime.maxComputeWorkGroupCountX;
    if (requiredGroupCount == 0u || deviceGroupLimit == 0u) {
        return 0u;
    }

    return static_cast<uint32_t>(
        std::min<uint64_t>(requiredGroupCount, deviceGroupLimit)
    );
}

uint32_t calculateDispatchWorkGroupCount(uint64_t invocationCount) {
    if (invocationCount == 0u) {
        return 0u;
    }

    const uint64_t requiredGroups =
        1u + (invocationCount - 1u) / kLocalWorkGroupSize;
    return calculateDispatchGroupCount(requiredGroups);
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
        if (gRuntime.submissionFence != VK_NULL_HANDLE) {
            vkDestroyFence(gRuntime.device, gRuntime.submissionFence, nullptr);
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
    int selectedMemoryScore = -1;

    // Prefer host-visible/device-local memory on UMA GPUs while preserving
    // coherent and cached-memory preferences for safe CPU readback.
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

        int score = 0;
        if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u) {
            score += 4;
        }
        if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0u) {
            score += 2;
        }
        if ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0u) {
            score += 1;
        }

        if (score > selectedMemoryScore) {
            selectedMemoryType = index;
            selectedMemoryScore = score;
            selectedMemoryIsCoherent =
                (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u;
        }
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
    const VkDeviceSize maximumRange =
        gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange;
    if (requiredBytes == 0u || requiredBytes > maximumRange) {
        return false;
    }

    if (gRuntime.storageBuffer != VK_NULL_HANDLE &&
        gRuntime.storageCapacity >= requiredBytes) {
        return true;
    }

    // Geometric growth amortizes allocation churn as tensor workload sizes change.
    // Fall back to the exact requested capacity when additional headroom cannot be allocated.
    if (gRuntime.storageCapacity > 0u) {
        const VkDeviceSize growth = std::max<VkDeviceSize>(
            gRuntime.storageCapacity / 2u,
            4096u
        );
        const VkDeviceSize remainingRange = maximumRange - gRuntime.storageCapacity;
        const VkDeviceSize grownCapacity = growth > remainingRange
            ? maximumRange
            : gRuntime.storageCapacity + growth;
        const VkDeviceSize targetCapacity =
            std::max(requiredBytes, grownCapacity);

        if (targetCapacity > requiredBytes &&
            createStorageBuffer(targetCapacity)) {
            return true;
        }
    }

    return createStorageBuffer(requiredBytes);
}

bool submitAndWait(const VkSubmitInfo& submitInfo, const char* operation) {
    if (gRuntime.device == VK_NULL_HANDLE ||
        gRuntime.computeQueue == VK_NULL_HANDLE ||
        gRuntime.submissionFence == VK_NULL_HANDLE) {
        logError(operation);
        return false;
    }

    VkResult result = vkResetFences(
        gRuntime.device,
        1,
        &gRuntime.submissionFence
    );
    if (result != VK_SUCCESS) {
        logError(operation, result);
        return false;
    }

    result = vkQueueSubmit(
        gRuntime.computeQueue,
        1,
        &submitInfo,
        gRuntime.submissionFence
    );
    if (result != VK_SUCCESS) {
        logError(operation, result);
        return false;
    }

    result = vkWaitForFences(
        gRuntime.device,
        1,
        &gRuntime.submissionFence,
        VK_TRUE,
        std::numeric_limits<uint64_t>::max()
    );
    if (result != VK_SUCCESS) {
        logError(operation, result);
        // Error-only recovery before caller releases or reuses mapped resources.
        const VkResult recoveryResult = vkQueueWaitIdle(gRuntime.computeQueue);
        if (recoveryResult != VK_SUCCESS) {
            logError("Vulkan queue-idle recovery after fence failure also failed.", recoveryResult);
        }
        return false;
    }

    return true;
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
    #if defined(VK_VERSION_1_3)
    gRuntime.instanceApiVersion = std::min(loaderApiVersion, VK_API_VERSION_1_3);
#elif defined(VK_VERSION_1_2)
    gRuntime.instanceApiVersion = std::min(loaderApiVersion, VK_API_VERSION_1_2);
#else
    gRuntime.instanceApiVersion = loaderApiVersion;
#endif
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

    uint32_t extensionCount = 0;
    VkResult extensionResult = vkEnumerateDeviceExtensionProperties(
        gRuntime.physicalDevice,
        nullptr,
        &extensionCount,
        nullptr
    );
    std::vector<VkExtensionProperties> deviceExtensions;
    if (extensionResult == VK_SUCCESS && extensionCount > 0u) {
        try {
            deviceExtensions.resize(extensionCount);
        } catch (const std::bad_alloc&) {
            logError("Not enough memory to enumerate Vulkan device extensions.");
            return false;
        }

        extensionResult = vkEnumerateDeviceExtensionProperties(
            gRuntime.physicalDevice,
            nullptr,
            &extensionCount,
            deviceExtensions.data()
        );
        if (extensionResult != VK_SUCCESS && extensionResult != VK_INCOMPLETE) {
            deviceExtensions.clear();
            logError("Vulkan device-extension enumeration failed.", extensionResult);
        } else {
            deviceExtensions.resize(extensionCount);
        }
    } else if (extensionResult != VK_SUCCESS) {
        logError("Vulkan device-extension count query failed.", extensionResult);
    }

    const uint32_t physicalApiVersion = gRuntime.physicalDeviceProperties.apiVersion;
    const bool physicalSupportsVulkan11 = physicalApiVersion >= VK_API_VERSION_1_1;
    const bool physicalSupportsVulkan12 = physicalApiVersion >= VK_API_VERSION_1_2;
#if defined(VK_VERSION_1_3)
    const bool physicalSupportsVulkan13 = physicalApiVersion >= VK_API_VERSION_1_3;
#else
    const bool physicalSupportsVulkan13 = false;
#endif

#if defined(VK_EXT_subgroup_size_control)
    const bool hasSubgroupSizeControlExtension = hasDeviceExtension(
        deviceExtensions,
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME
    );
#else
    const bool hasSubgroupSizeControlExtension = false;
#endif

#if defined(VK_KHR_shader_integer_dot_product)
    const bool hasIntegerDotProductExtension = hasDeviceExtension(
        deviceExtensions,
        VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME
    );
#else
    const bool hasIntegerDotProductExtension = false;
#endif

#if defined(VK_KHR_cooperative_matrix)
    gRuntime.hasCooperativeMatrixExtension = hasDeviceExtension(
        deviceExtensions,
        VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
    );
#endif

    // Query extension and promoted-core features before creating the logical device.
    // Only the existing FP16 path is enabled today; other feature flags are recorded
    // for later pipeline selection and are deliberately not enabled speculatively.
    VkPhysicalDeviceFeatures2 supportedFeatures{};
    supportedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    void** featureChainTail = &supportedFeatures.pNext;

    auto appendFeature = [&featureChainTail](auto& feature) {
        feature.pNext = nullptr;
        *featureChainTail = &feature;
        featureChainTail = &feature.pNext;
    };

    VkPhysicalDeviceShaderFloat16Int8Features supportedShaderFloat16{};
    // The current FP16 kernel is compiled for the Vulkan 1.2 core path.
    // Do not activate the extension-only feature on Vulkan 1.1 devices until
    // the extension is explicitly enabled in VkDeviceCreateInfo.
    const bool queryShaderFloat16 = physicalSupportsVulkan12;
    if (queryShaderFloat16) {
        supportedShaderFloat16.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        appendFeature(supportedShaderFloat16);
    }

#if defined(VK_VERSION_1_1)
    VkPhysicalDevice16BitStorageFeatures supportedStorage16{};
    if (physicalSupportsVulkan11) {
        supportedStorage16.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        appendFeature(supportedStorage16);
    }
#endif

#if defined(VK_VERSION_1_3)
    VkPhysicalDeviceSubgroupSizeControlFeatures supportedSubgroupSizeControl{};
    const bool querySubgroupSizeControl =
        physicalSupportsVulkan13 || hasSubgroupSizeControlExtension;
    if (querySubgroupSizeControl) {
        supportedSubgroupSizeControl.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
        appendFeature(supportedSubgroupSizeControl);
    }

    VkPhysicalDeviceShaderIntegerDotProductFeatures supportedIntegerDotProduct{};
    const bool queryIntegerDotProduct =
        physicalSupportsVulkan13 || hasIntegerDotProductExtension;
    if (queryIntegerDotProduct) {
        supportedIntegerDotProduct.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
        appendFeature(supportedIntegerDotProduct);
    }
#else
    const bool querySubgroupSizeControl = hasSubgroupSizeControlExtension;
    const bool queryIntegerDotProduct = hasIntegerDotProductExtension;
#if defined(VK_EXT_subgroup_size_control)
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT supportedSubgroupSizeControl{};
    if (querySubgroupSizeControl) {
        supportedSubgroupSizeControl.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
        appendFeature(supportedSubgroupSizeControl);
    }
#endif
#if defined(VK_KHR_shader_integer_dot_product)
    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR supportedIntegerDotProduct{};
    if (queryIntegerDotProduct) {
        supportedIntegerDotProduct.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
        appendFeature(supportedIntegerDotProduct);
    }
#endif
#endif

#if defined(VK_KHR_cooperative_matrix)
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR supportedCooperativeMatrix{};
    if (gRuntime.hasCooperativeMatrixExtension) {
        supportedCooperativeMatrix.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        appendFeature(supportedCooperativeMatrix);
    }
#endif

    const auto getPhysicalDeviceFeatures2 =
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
            vkGetInstanceProcAddr(gRuntime.instance, "vkGetPhysicalDeviceFeatures2")
        );
    const bool features2Available =
        gRuntime.instanceApiVersion >= VK_API_VERSION_1_1 &&
        physicalSupportsVulkan11 &&
        getPhysicalDeviceFeatures2 != nullptr;

    if (features2Available) {
        getPhysicalDeviceFeatures2(gRuntime.physicalDevice, &supportedFeatures);
        gRuntime.supportsFp16Compute =
            queryShaderFloat16 && supportedShaderFloat16.shaderFloat16 == VK_TRUE;
        gRuntime.supportsShaderInt8 =
            queryShaderFloat16 && supportedShaderFloat16.shaderInt8 == VK_TRUE;
#if defined(VK_VERSION_1_1)
        gRuntime.supportsStorageBuffer16BitAccess =
            physicalSupportsVulkan11 &&
            supportedStorage16.storageBuffer16BitAccess == VK_TRUE;
#endif
#if defined(VK_VERSION_1_3)
        gRuntime.supportsSubgroupSizeControl =
            querySubgroupSizeControl &&
            supportedSubgroupSizeControl.subgroupSizeControl == VK_TRUE;
        gRuntime.supportsComputeFullSubgroups =
            querySubgroupSizeControl &&
            supportedSubgroupSizeControl.computeFullSubgroups == VK_TRUE;
        gRuntime.supportsShaderIntegerDotProduct =
            queryIntegerDotProduct &&
            supportedIntegerDotProduct.shaderIntegerDotProduct == VK_TRUE;
#else
#if defined(VK_EXT_subgroup_size_control)
        gRuntime.supportsSubgroupSizeControl =
            querySubgroupSizeControl &&
            supportedSubgroupSizeControl.subgroupSizeControl == VK_TRUE;
        gRuntime.supportsComputeFullSubgroups =
            querySubgroupSizeControl &&
            supportedSubgroupSizeControl.computeFullSubgroups == VK_TRUE;
#endif
#if defined(VK_KHR_shader_integer_dot_product)
        gRuntime.supportsShaderIntegerDotProduct =
            queryIntegerDotProduct &&
            supportedIntegerDotProduct.shaderIntegerDotProduct == VK_TRUE;
#endif
#endif
#if defined(VK_KHR_cooperative_matrix)
        gRuntime.supportsCooperativeMatrix =
            gRuntime.hasCooperativeMatrixExtension &&
            supportedCooperativeMatrix.cooperativeMatrix == VK_TRUE;
        gRuntime.supportsCooperativeMatrixRobustBufferAccess =
            gRuntime.hasCooperativeMatrixExtension &&
            supportedCooperativeMatrix.cooperativeMatrixRobustBufferAccess == VK_TRUE;
#endif
    } else {
        gRuntime.supportsFp16Compute = false;
        gRuntime.supportsShaderInt8 = false;
    }

    // Properties are queried separately from features. A supported bit is not
    // treated as accelerated hardware unless the matching property says so.
    const auto getPhysicalDeviceProperties2 =
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            vkGetInstanceProcAddr(gRuntime.instance, "vkGetPhysicalDeviceProperties2")
        );
    const bool properties2Available =
        gRuntime.instanceApiVersion >= VK_API_VERSION_1_1 &&
        physicalSupportsVulkan11 &&
        getPhysicalDeviceProperties2 != nullptr;

    if (properties2Available) {
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        void** propertyChainTail = &properties2.pNext;

        auto appendProperty = [&propertyChainTail](auto& property) {
            property.pNext = nullptr;
            *propertyChainTail = &property;
            propertyChainTail = &property.pNext;
        };

        VkPhysicalDeviceSubgroupProperties subgroupProperties{};
        subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        appendProperty(subgroupProperties);

        VkPhysicalDeviceDriverProperties driverProperties{};
        if (physicalSupportsVulkan12) {
            driverProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
            appendProperty(driverProperties);
        }

#if defined(VK_VERSION_1_3)
        VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeProperties{};
        if (querySubgroupSizeControl) {
            subgroupSizeProperties.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
            appendProperty(subgroupSizeProperties);
        }

        VkPhysicalDeviceShaderIntegerDotProductProperties integerDotProductProperties{};
        if (queryIntegerDotProduct) {
            integerDotProductProperties.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
            appendProperty(integerDotProductProperties);
        }
#else
#if defined(VK_EXT_subgroup_size_control)
        VkPhysicalDeviceSubgroupSizeControlPropertiesEXT subgroupSizeProperties{};
        if (querySubgroupSizeControl) {
            subgroupSizeProperties.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
            appendProperty(subgroupSizeProperties);
        }
#endif
#if defined(VK_KHR_shader_integer_dot_product)
        VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR integerDotProductProperties{};
        if (queryIntegerDotProduct) {
            integerDotProductProperties.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR;
            appendProperty(integerDotProductProperties);
        }
#endif
#endif

#if defined(VK_KHR_cooperative_matrix)
        VkPhysicalDeviceCooperativeMatrixPropertiesKHR cooperativeMatrixProperties{};
        if (gRuntime.hasCooperativeMatrixExtension) {
            cooperativeMatrixProperties.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            appendProperty(cooperativeMatrixProperties);
        }
#endif

        getPhysicalDeviceProperties2(gRuntime.physicalDevice, &properties2);
        gRuntime.physicalDeviceProperties = properties2.properties;
        gRuntime.subgroupSize = subgroupProperties.subgroupSize;
        gRuntime.subgroupSupportedStages = subgroupProperties.supportedStages;
        gRuntime.subgroupSupportedOperations = subgroupProperties.supportedOperations;
        gRuntime.subgroupQuadOperationsInAllStages =
            subgroupProperties.quadOperationsInAllStages == VK_TRUE;

        if (physicalSupportsVulkan12) {
            gRuntime.driverProperties = driverProperties;
            gRuntime.hasDriverProperties =
                driverProperties.driverName[0] != '\0' ||
                driverProperties.driverInfo[0] != '\0';
        }

#if defined(VK_VERSION_1_3)
        if (querySubgroupSizeControl) {
            gRuntime.minSubgroupSize = subgroupSizeProperties.minSubgroupSize;
            gRuntime.maxSubgroupSize = subgroupSizeProperties.maxSubgroupSize;
            gRuntime.maxComputeWorkgroupSubgroups =
                subgroupSizeProperties.maxComputeWorkgroupSubgroups;
            gRuntime.requiredSubgroupSizeStages =
                subgroupSizeProperties.requiredSubgroupSizeStages;
        }
        if (queryIntegerDotProduct) {
            gRuntime.integerDotProduct8BitUnsignedAccelerated =
                integerDotProductProperties.integerDotProduct8BitUnsignedAccelerated == VK_TRUE;
            gRuntime.integerDotProduct8BitSignedAccelerated =
                integerDotProductProperties.integerDotProduct8BitSignedAccelerated == VK_TRUE;
            gRuntime.integerDotProduct8BitMixedSignednessAccelerated =
                integerDotProductProperties.integerDotProduct8BitMixedSignednessAccelerated == VK_TRUE;
            gRuntime.integerDotProduct4x8BitPackedUnsignedAccelerated =
                integerDotProductProperties.integerDotProduct4x8BitPackedUnsignedAccelerated == VK_TRUE;
            gRuntime.integerDotProduct4x8BitPackedSignedAccelerated =
                integerDotProductProperties.integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;
            gRuntime.integerDotProduct4x8BitPackedMixedSignednessAccelerated =
                integerDotProductProperties.integerDotProduct4x8BitPackedMixedSignednessAccelerated == VK_TRUE;
        }
#else
#if defined(VK_EXT_subgroup_size_control)
        if (querySubgroupSizeControl) {
            gRuntime.minSubgroupSize = subgroupSizeProperties.minSubgroupSize;
            gRuntime.maxSubgroupSize = subgroupSizeProperties.maxSubgroupSize;
            gRuntime.maxComputeWorkgroupSubgroups =
                subgroupSizeProperties.maxComputeWorkgroupSubgroups;
            gRuntime.requiredSubgroupSizeStages =
                subgroupSizeProperties.requiredSubgroupSizeStages;
        }
#endif
#if defined(VK_KHR_shader_integer_dot_product)
        if (queryIntegerDotProduct) {
            gRuntime.integerDotProduct8BitUnsignedAccelerated =
                integerDotProductProperties.integerDotProduct8BitUnsignedAccelerated == VK_TRUE;
            gRuntime.integerDotProduct8BitSignedAccelerated =
                integerDotProductProperties.integerDotProduct8BitSignedAccelerated == VK_TRUE;
            gRuntime.integerDotProduct8BitMixedSignednessAccelerated =
                integerDotProductProperties.integerDotProduct8BitMixedSignednessAccelerated == VK_TRUE;
            gRuntime.integerDotProduct4x8BitPackedUnsignedAccelerated =
                integerDotProductProperties.integerDotProduct4x8BitPackedUnsignedAccelerated == VK_TRUE;
            gRuntime.integerDotProduct4x8BitPackedSignedAccelerated =
                integerDotProductProperties.integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;
            gRuntime.integerDotProduct4x8BitPackedMixedSignednessAccelerated =
                integerDotProductProperties.integerDotProduct4x8BitPackedMixedSignednessAccelerated == VK_TRUE;
        }
#endif
#endif

#if defined(VK_KHR_cooperative_matrix)
        if (gRuntime.hasCooperativeMatrixExtension) {
            gRuntime.cooperativeMatrixSupportedStages =
                cooperativeMatrixProperties.cooperativeMatrixSupportedStages;
        }
#endif
    }

#if defined(VK_KHR_cooperative_matrix)
    if (gRuntime.supportsCooperativeMatrix) {
        const auto getCooperativeMatrixProperties =
            reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
                vkGetInstanceProcAddr(
                    gRuntime.instance,
                    "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"
                )
            );
        if (getCooperativeMatrixProperties != nullptr) {
            uint32_t matrixPropertyCount = 0;
            VkResult matrixResult = getCooperativeMatrixProperties(
                gRuntime.physicalDevice,
                &matrixPropertyCount,
                nullptr
            );
            if (matrixResult == VK_SUCCESS && matrixPropertyCount > 0u) {
                std::vector<VkCooperativeMatrixPropertiesKHR> matrixProperties;
                try {
                    matrixProperties.resize(matrixPropertyCount);
                } catch (const std::bad_alloc&) {
                    logError("Not enough memory to enumerate cooperative-matrix configurations.");
                    return false;
                }

                for (auto& matrixProperty : matrixProperties) {
                    matrixProperty.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                }

                matrixResult = getCooperativeMatrixProperties(
                    gRuntime.physicalDevice,
                    &matrixPropertyCount,
                    matrixProperties.data()
                );
                if (matrixResult == VK_SUCCESS || matrixResult == VK_INCOMPLETE) {
                    matrixProperties.resize(matrixPropertyCount);
                    gRuntime.cooperativeMatrixTypes.reserve(matrixProperties.size());
                    for (const auto& matrixProperty : matrixProperties) {
                        gRuntime.cooperativeMatrixTypes.push_back({
                            matrixProperty.MSize,
                            matrixProperty.NSize,
                            matrixProperty.KSize,
                            static_cast<uint32_t>(matrixProperty.AType),
                            static_cast<uint32_t>(matrixProperty.BType),
                            static_cast<uint32_t>(matrixProperty.CType),
                            static_cast<uint32_t>(matrixProperty.ResultType),
                            matrixProperty.saturatingAccumulation == VK_TRUE,
                            static_cast<uint32_t>(matrixProperty.scope)
                        });
                    }
                } else {
                    logError("Cooperative-matrix configuration query failed.", matrixResult);
                }
            } else if (matrixResult != VK_SUCCESS) {
                logError("Cooperative-matrix configuration count query failed.", matrixResult);
            }
        }
    }
#endif

    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = gRuntime.queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures deviceFeatures{};
    VkPhysicalDeviceShaderFloat16Int8Features enabledShaderFloat16{};
    VkPhysicalDeviceFeatures2 enabledFeatures{};

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    if (gRuntime.supportsFp16Compute) {
        enabledShaderFloat16.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        enabledShaderFloat16.shaderFloat16 = VK_TRUE;
        enabledFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        enabledFeatures.pNext = &enabledShaderFloat16;
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

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    result = vkCreateFence(
        gRuntime.device,
        &fenceInfo,
        nullptr,
        &gRuntime.submissionFence
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan submission-fence creation failed.", result);
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

    const VkDeviceSize maxStorageBufferRange =
        gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange;
    constexpr size_t kAdamwTensorArrayCount = 4u;
    constexpr size_t kAdamwValueSize = sizeof(float);
    constexpr size_t kAdamwBytesPerElement =
        kAdamwTensorArrayCount * kAdamwValueSize;
    const VkDeviceSize maxPageElementsDevice =
        maxStorageBufferRange / kAdamwBytesPerElement;
    if (maxPageElementsDevice == 0u ||
        maxPageElementsDevice > std::numeric_limits<size_t>::max()) {
        logError("Vulkan storage-buffer range cannot hold one AdamW element page.");
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }
    const size_t maxPageElements =
        static_cast<size_t>(maxPageElementsDevice);

    // Keep the transaction atomic with respect to model state. GPU work is
    // performed one storage-buffer page at a time, but no Java tensor is
    // changed until every page has completed and every result is finite.
    std::vector<float> stagedParameters;
    std::vector<float> stagedFirstMoments;
    std::vector<float> stagedSecondMoments;
    try {
        stagedParameters.resize(totalElements);
        stagedFirstMoments.resize(totalElements);
        stagedSecondMoments.resize(totalElements);
    } catch (const std::bad_alloc&) {
        logError("Not enough host memory to retain the complete staged AdamW result.");
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }

    VkResult result = VK_SUCCESS;
    size_t pageOffset = 0u;
    size_t pageIndex = 0u;
    const size_t pageCount =
        1u + (totalElements - 1u) / maxPageElements;

    while (pageOffset < totalElements) {
        const size_t pageElementCount =
            std::min(maxPageElements, totalElements - pageOffset);
        const size_t pageEnd = pageOffset + pageElementCount;
        const VkDeviceSize pageByteCount =
            static_cast<VkDeviceSize>(pageElementCount) *
            static_cast<VkDeviceSize>(kAdamwBytesPerElement);

        if (pageElementCount == 0u ||
            pageElementCount > std::numeric_limits<uint32_t>::max() ||
            !ensureStorageBuffer(pageByteCount)) {
            logError("Vulkan AdamW storage-buffer page allocation failed.");
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }

        // The page layout is [parameters | gradients | first moments |
        // second moments]. Copy only each group's overlap with this page.
        float* parameterValues = static_cast<float*>(gRuntime.mappedStorage);
        float* gradientValues = parameterValues + pageElementCount;
        float* firstMomentValues = gradientValues + pageElementCount;
        float* secondMomentValues = firstMomentValues + pageElementCount;

        for (const GroupArrays& group : groups) {
            const size_t groupStart = group.packedOffset;
            const size_t groupEnd =
                groupStart + static_cast<size_t>(group.elementCount);
            const size_t overlapStart = std::max(groupStart, pageOffset);
            const size_t overlapEnd = std::min(groupEnd, pageEnd);
            if (overlapStart >= overlapEnd) {
                continue;
            }

            const size_t sourceOffset = overlapStart - groupStart;
            const size_t destinationOffset = overlapStart - pageOffset;
            const size_t overlapCount = overlapEnd - overlapStart;
            if (sourceOffset > static_cast<size_t>(group.elementCount) ||
                overlapCount > static_cast<size_t>(group.elementCount) - sourceOffset ||
                overlapCount > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
                logError("Vulkan AdamW page/group overlap is outside the source tensor.");
                releaseGroupReferences(env, groups);
                return JNI_FALSE;
            }

            const jsize sourceIndex = static_cast<jsize>(sourceOffset);
            const jsize copyCount = static_cast<jsize>(overlapCount);
            const size_t targetIndex = destinationOffset;

            env->GetFloatArrayRegion(
                group.parameters,
                sourceIndex,
                copyCount,
                parameterValues + targetIndex
            );
            if (env->ExceptionCheck()) {
                releaseGroupReferences(env, groups);
                return JNI_FALSE;
            }

            env->GetFloatArrayRegion(
                group.gradients,
                sourceIndex,
                copyCount,
                gradientValues + targetIndex
            );
            if (env->ExceptionCheck()) {
                releaseGroupReferences(env, groups);
                return JNI_FALSE;
            }

            env->GetFloatArrayRegion(
                group.firstMoments,
                sourceIndex,
                copyCount,
                firstMomentValues + targetIndex
            );
            if (env->ExceptionCheck()) {
                releaseGroupReferences(env, groups);
                return JNI_FALSE;
            }

            env->GetFloatArrayRegion(
                group.secondMoments,
                sourceIndex,
                copyCount,
                secondMomentValues + targetIndex
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
                logError("Failed to flush Vulkan AdamW page input memory.", flushResult);
                releaseGroupReferences(env, groups);
                return JNI_FALSE;
            }
        }

        result = vkResetCommandBuffer(gRuntime.commandBuffer, 0);
        if (result != VK_SUCCESS) {
            logError("Failed to reset the Vulkan AdamW page command buffer.", result);
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(gRuntime.commandBuffer, &beginInfo);
        if (result != VK_SUCCESS) {
            logError("Failed to begin the Vulkan AdamW page command buffer.", result);
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

        size_t dispatchedSegmentCount = 0u;
        for (const AdamwSegment& segment : segments) {
            const size_t segmentStart = segment.elementOffset;
            const size_t segmentEnd =
                segmentStart + static_cast<size_t>(segment.elementCount);
            const size_t overlapStart = std::max(segmentStart, pageOffset);
            const size_t overlapEnd = std::min(segmentEnd, pageEnd);
            if (overlapStart >= overlapEnd) {
                continue;
            }

            const size_t localOffset = overlapStart - pageOffset;
            const size_t localCount = overlapEnd - overlapStart;
            if (localOffset > std::numeric_limits<uint32_t>::max() ||
                localCount == 0u ||
                localCount > std::numeric_limits<uint32_t>::max()) {
                logError("Vulkan AdamW page segment exceeds the shader index range.");
                releaseGroupReferences(env, groups);
                return JNI_FALSE;
            }

            const uint32_t workGroupCount =
                calculateDispatchWorkGroupCount(localCount);
            if (workGroupCount == 0u) {
                logError("Vulkan AdamW page segment has no available work-groups.");
                releaseGroupReferences(env, groups);
                return JNI_FALSE;
            }

            const AdamwPushConstants pushConstants{
                static_cast<uint32_t>(pageElementCount),
                static_cast<uint32_t>(localOffset),
                static_cast<uint32_t>(localCount),
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
            vkCmdDispatch(
                gRuntime.commandBuffer,
                workGroupCount,
                1,
                1
            );
            ++dispatchedSegmentCount;
        }

        if (dispatchedSegmentCount == 0u) {
            logError("Vulkan AdamW page did not intersect an optimizer segment.");
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
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
            logError("Failed to finish the Vulkan AdamW page command buffer.", result);
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &gRuntime.commandBuffer;
        // The mapped page is read back before reuse, so completion remains synchronous.
        // A fence targets this dispatch instead of idling the entire queue.
        if (!submitAndWait(
                submitInfo,
                "Vulkan AdamW page submission or fence wait failed."
            )) {
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
                logError("Failed to invalidate Vulkan AdamW page output memory.", result);
                releaseGroupReferences(env, groups);
                return JNI_FALSE;
            }
        }

        const float* pageOutputParameters =
            static_cast<const float*>(gRuntime.mappedStorage);
        const float* pageOutputFirstMoments =
            pageOutputParameters + pageElementCount * 2u;
        const float* pageOutputSecondMoments =
            pageOutputParameters + pageElementCount * 3u;

        for (size_t index = 0u; index < pageElementCount; ++index) {
            if (!std::isfinite(pageOutputParameters[index]) ||
                !std::isfinite(pageOutputFirstMoments[index]) ||
                !std::isfinite(pageOutputSecondMoments[index])) {
                logError("Vulkan AdamW produced a non-finite value in a storage-buffer page.");
                releaseGroupReferences(env, groups);
                return JNI_FALSE;
            }
        }

        std::copy_n(
            pageOutputParameters,
            pageElementCount,
            stagedParameters.data() + pageOffset
        );
        std::copy_n(
            pageOutputFirstMoments,
            pageElementCount,
            stagedFirstMoments.data() + pageOffset
        );
        std::copy_n(
            pageOutputSecondMoments,
            pageElementCount,
            stagedSecondMoments.data() + pageOffset
        );

        pageOffset = pageEnd;
        ++pageIndex;
    }

    if (pageIndex != pageCount) {
        logError("Vulkan AdamW did not process every storage-buffer page.");
        releaseGroupReferences(env, groups);
        return JNI_FALSE;
    }

    // Validate the complete result before mutating any Java model/optimizer
    // arrays. Earlier pages remain only in host staging if a later page fails.
    for (size_t index = 0u; index < totalElements; ++index) {
        if (!std::isfinite(stagedParameters[index]) ||
            !std::isfinite(stagedFirstMoments[index]) ||
            !std::isfinite(stagedSecondMoments[index])) {
            logError("Vulkan AdamW staged output contains a non-finite value.");
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
            stagedParameters.data() + offset
        );
        if (env->ExceptionCheck()) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }

        env->SetFloatArrayRegion(
            group.firstMoments,
            0,
            count,
            stagedFirstMoments.data() + offset
        );
        if (env->ExceptionCheck()) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }

        env->SetFloatArrayRegion(
            group.secondMoments,
            0,
            count,
            stagedSecondMoments.data() + offset
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

    // Each 8x16 workgroup computes one output tile. The shader advances
    // over additional tiles by gl_NumWorkGroups.x, so dispatch remains within
    // the device X limit without rejecting large, otherwise valid matrices.
    constexpr uint64_t kLinearTileRows = 8u;
    constexpr uint64_t kLinearTileColumns = 16u;
    const uint64_t tileRowsPerBatch =
        1u + (static_cast<uint64_t>(rowsPerBatch) - 1u) / kLinearTileRows;
    const uint64_t tileColumnsPerBatch =
        1u + (static_cast<uint64_t>(outputColumns) - 1u) / kLinearTileColumns;
    const uint64_t totalTileCount =
        static_cast<uint64_t>(batchCount) *
        tileRowsPerBatch *
        tileColumnsPerBatch;
    const uint32_t workGroupCount =
        calculateDispatchGroupCount(totalTileCount);
    if (workGroupCount == 0u) {
        logError("Vulkan linear dispatch has no available output tile groups.");
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

    if (!submitAndWait(
            submitInfo,
            "Vulkan linear submission or fence wait failed."
        )) {
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

    // Convert FP32 activations directly into the mapped packed-half staging
    // buffer in bounded chunks. This avoids a second full-size host tensor
    // allocation while keeping JNI calls large enough to amortize call overhead.
    constexpr size_t kInputConversionChunkElements = 8192u;
    std::array<jfloat, kInputConversionChunkElements> inputChunk{};
    auto* packedWords = static_cast<uint32_t*>(gRuntime.mappedStorage);

    for (uint64_t chunkOffset = 0u;
         chunkOffset < inputElements;
         chunkOffset += kInputConversionChunkElements) {
        const size_t chunkCount = static_cast<size_t>(
            std::min<uint64_t>(
                kInputConversionChunkElements,
                inputElements - chunkOffset
            )
        );
        env->GetFloatArrayRegion(
            inputArray,
            static_cast<jsize>(chunkOffset),
            static_cast<jsize>(chunkCount),
            inputChunk.data()
        );
        if (env->ExceptionCheck()) {
            return JNI_FALSE;
        }

        // Chunk size and offsets are even, so only the final chunk can leave
        // one high-half slot unused. That slot is explicitly zero-filled.
        for (size_t localIndex = 0u; localIndex < chunkCount; localIndex += 2u) {
            const size_t absoluteIndex =
                static_cast<size_t>(chunkOffset) + localIndex;
            const float firstValue = inputChunk[localIndex];
            if (!std::isfinite(firstValue)) {
                return JNI_FALSE;
            }

            const uint16_t lowHalf = floatToHalf(firstValue);
            if ((lowHalf & 0x7C00u) == 0x7C00u) {
                logError("FP16 input activation exceeds the finite binary16 range.");
                return JNI_FALSE;
            }

            uint32_t highHalf = 0u;
            if (localIndex + 1u < chunkCount) {
                const float secondValue = inputChunk[localIndex + 1u];
                if (!std::isfinite(secondValue)) {
                    return JNI_FALSE;
                }
                const uint16_t encodedSecond = floatToHalf(secondValue);
                if ((encodedSecond & 0x7C00u) == 0x7C00u) {
                    logError("FP16 input activation exceeds the finite binary16 range.");
                    return JNI_FALSE;
                }
                highHalf = static_cast<uint32_t>(encodedSecond) << 16u;
            }

            packedWords[absoluteIndex >> 1u] =
                static_cast<uint32_t>(lowHalf) | highHalf;
        }
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

    // Each 8x16 workgroup computes one FP16 output tile. Additional tiles
    // are covered by the shader's workgroup grid-stride loop.
    constexpr uint64_t kFp16TileRows = 8u;
    constexpr uint64_t kFp16TileColumns = 16u;
    const uint64_t tileRowsPerBatch =
        1u + (static_cast<uint64_t>(rowsPerBatch) - 1u) / kFp16TileRows;
    const uint64_t tileColumnsPerBatch =
        1u + (static_cast<uint64_t>(outputColumns) - 1u) / kFp16TileColumns;
    const uint64_t totalTileCount =
        static_cast<uint64_t>(batchCount) *
        tileRowsPerBatch *
        tileColumnsPerBatch;
    const uint32_t workGroupCount =
        calculateDispatchGroupCount(totalTileCount);
    if (workGroupCount == 0u) {
        logError("Vulkan FP16 linear dispatch has no available output tile groups.");
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
    if (!submitAndWait(
            submitInfo,
            "Vulkan FP16 linear submission or fence wait failed."
        )) {
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
    for (uint64_t index = 0u; index < outputElements; ++index) {
        const uint32_t valueBits = mappedWords[outputOffsetWords + index];
        if ((valueBits & 0x7F800000u) == 0x7F800000u) {
            logError("Vulkan FP16 linear kernel produced a non-finite output.");
            return JNI_FALSE;
        }
    }
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

extern "C" JNIEXPORT jstring JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeGetDriverInfo(
    JNIEnv* env,
    jobject
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    if (gRuntime.physicalDevice == VK_NULL_HANDLE) {
        return env->NewStringUTF("Vulkan driver unavailable");
    }

    const VkPhysicalDeviceProperties& properties =
        gRuntime.physicalDeviceProperties;
    char driverInfo[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE +
                    VK_MAX_DRIVER_NAME_SIZE +
                    VK_MAX_DRIVER_INFO_SIZE + 32]{};

    if (gRuntime.hasDriverProperties) {
        const char* driverName =
            gRuntime.driverProperties.driverName[0] != '\0'
                ? gRuntime.driverProperties.driverName
                : "Vulkan driver";
        const char* details =
            gRuntime.driverProperties.driverInfo[0] != '\0'
                ? gRuntime.driverProperties.driverInfo
                : "version details unavailable";
        std::snprintf(
            driverInfo,
            sizeof(driverInfo),
            "%s | %s: %s",
            properties.deviceName,
            driverName,
            details
        );
    } else {
        std::snprintf(
            driverInfo,
            sizeof(driverInfo),
            "%s | vendor-defined driver version 0x%08X",
            properties.deviceName,
            properties.driverVersion
        );
    }
    return env->NewStringUTF(driverInfo);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeGetDeviceCapabilities(
    JNIEnv* env,
    jobject
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    if (gRuntime.physicalDevice == VK_NULL_HANDLE) {
        return env->NewStringUTF("{}");
    }

    const VkPhysicalDeviceProperties& properties = gRuntime.physicalDeviceProperties;
    std::ostringstream json;
    json << "{"
         << "\"deviceName\":\"" << escapeJsonString(properties.deviceName) << "\","
         << "\"vendorId\":" << properties.vendorID << ","
         << "\"deviceId\":" << properties.deviceID << ","
         << "\"deviceType\":" << static_cast<uint32_t>(properties.deviceType) << ","
         << "\"apiVersionMajor\":" << VK_VERSION_MAJOR(properties.apiVersion) << ","
         << "\"apiVersionMinor\":" << VK_VERSION_MINOR(properties.apiVersion) << ","
         << "\"apiVersionPatch\":" << VK_VERSION_PATCH(properties.apiVersion) << ","
         << "\"driverName\":\"" << escapeJsonString(
                gRuntime.hasDriverProperties ? gRuntime.driverProperties.driverName : ""
            ) << "\","
         << "\"driverInfo\":\"" << escapeJsonString(
                gRuntime.hasDriverProperties ? gRuntime.driverProperties.driverInfo : ""
            ) << "\","
         << "\"driverVersionRaw\":" << properties.driverVersion << ","
         << "\"maxComputeWorkGroupInvocations\":"
         << properties.limits.maxComputeWorkGroupInvocations << ","
         << "\"maxComputeWorkGroupSizeX\":"
         << properties.limits.maxComputeWorkGroupSize[0] << ","
         << "\"maxComputeWorkGroupSizeY\":"
         << properties.limits.maxComputeWorkGroupSize[1] << ","
         << "\"maxComputeWorkGroupSizeZ\":"
         << properties.limits.maxComputeWorkGroupSize[2] << ","
         << "\"maxComputeWorkGroupCountX\":"
         << properties.limits.maxComputeWorkGroupCount[0] << ","
         << "\"maxComputeWorkGroupCountY\":"
         << properties.limits.maxComputeWorkGroupCount[1] << ","
         << "\"maxComputeWorkGroupCountZ\":"
         << properties.limits.maxComputeWorkGroupCount[2] << ","
         << "\"maxComputeSharedMemoryBytes\":"
         << properties.limits.maxComputeSharedMemorySize << ","
         << "\"maxStorageBufferRangeBytes\":"
         << properties.limits.maxStorageBufferRange << ","
         << "\"subgroupSize\":" << gRuntime.subgroupSize << ","
         << "\"subgroupSupportedStages\":" << gRuntime.subgroupSupportedStages << ","
         << "\"subgroupSupportedOperations\":" << gRuntime.subgroupSupportedOperations << ","
         << "\"subgroupQuadOperationsInAllStages\":"
         << (gRuntime.subgroupQuadOperationsInAllStages ? "true" : "false") << ","
         << "\"subgroupSizeControl\":"
         << (gRuntime.supportsSubgroupSizeControl ? "true" : "false") << ","
         << "\"computeFullSubgroups\":"
         << (gRuntime.supportsComputeFullSubgroups ? "true" : "false") << ","
         << "\"minSubgroupSize\":" << gRuntime.minSubgroupSize << ","
         << "\"maxSubgroupSize\":" << gRuntime.maxSubgroupSize << ","
         << "\"maxComputeWorkgroupSubgroups\":"
         << gRuntime.maxComputeWorkgroupSubgroups << ","
         << "\"requiredSubgroupSizeStages\":" << gRuntime.requiredSubgroupSizeStages << ","
         << "\"shaderFloat16\":" << (gRuntime.supportsFp16Compute ? "true" : "false") << ","
         << "\"shaderInt8\":" << (gRuntime.supportsShaderInt8 ? "true" : "false") << ","
         << "\"storageBuffer16BitAccess\":"
         << (gRuntime.supportsStorageBuffer16BitAccess ? "true" : "false") << ","
         << "\"shaderIntegerDotProduct\":"
         << (gRuntime.supportsShaderIntegerDotProduct ? "true" : "false") << ","
         << "\"integerDotProduct8BitUnsignedAccelerated\":"
         << (gRuntime.integerDotProduct8BitUnsignedAccelerated ? "true" : "false") << ","
         << "\"integerDotProduct8BitSignedAccelerated\":"
         << (gRuntime.integerDotProduct8BitSignedAccelerated ? "true" : "false") << ","
         << "\"integerDotProduct8BitMixedSignednessAccelerated\":"
         << (gRuntime.integerDotProduct8BitMixedSignednessAccelerated ? "true" : "false") << ","
         << "\"integerDotProduct4x8BitPackedUnsignedAccelerated\":"
         << (gRuntime.integerDotProduct4x8BitPackedUnsignedAccelerated ? "true" : "false") << ","
         << "\"integerDotProduct4x8BitPackedSignedAccelerated\":"
         << (gRuntime.integerDotProduct4x8BitPackedSignedAccelerated ? "true" : "false") << ","
         << "\"integerDotProduct4x8BitPackedMixedSignednessAccelerated\":"
         << (gRuntime.integerDotProduct4x8BitPackedMixedSignednessAccelerated ? "true" : "false") << ","
         << "\"cooperativeMatrixExtension\":"
         << (gRuntime.hasCooperativeMatrixExtension ? "true" : "false") << ","
         << "\"cooperativeMatrix\":"
         << (gRuntime.supportsCooperativeMatrix ? "true" : "false") << ","
         << "\"cooperativeMatrixRobustBufferAccess\":"
         << (gRuntime.supportsCooperativeMatrixRobustBufferAccess ? "true" : "false") << ","
         << "\"cooperativeMatrixSupportedStages\":"
         << gRuntime.cooperativeMatrixSupportedStages << ","
         << "\"cooperativeMatrixTypes\":[";

    for (size_t index = 0; index < gRuntime.cooperativeMatrixTypes.size(); ++index) {
        const auto& matrix = gRuntime.cooperativeMatrixTypes[index];
        if (index > 0u) {
            json << ",";
        }
        json << "{"
             << "\"mSize\":" << matrix.mSize << ","
             << "\"nSize\":" << matrix.nSize << ","
             << "\"kSize\":" << matrix.kSize << ","
             << "\"aType\":" << matrix.aType << ","
             << "\"bType\":" << matrix.bType << ","
             << "\"cType\":" << matrix.cType << ","
             << "\"resultType\":" << matrix.resultType << ","
             << "\"saturatingAccumulation\":"
             << (matrix.saturatingAccumulation ? "true" : "false") << ","
             << "\"scope\":" << matrix.scope
             << "}";
    }

    json << "]}";
    const std::string payload = json.str();
    return env->NewStringUTF(payload.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeDestroy(
    JNIEnv*,
    jobject
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    destroyRuntime();
}
