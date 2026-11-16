#include <jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>

#if !defined(VK_KHR_synchronization2)
#error "Argus requires Vulkan headers with VK_KHR_synchronization2 support."
#endif

#include "adamw_shader_spv.h"
#include "linear_shader_spv.h"
#include "linear_fp16_shader_spv.h"
#include "linear_resident_shader_spv.h"
#include "attention_resident_shader_spv.h"
#include "tensor_arena_core.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <initializer_list>
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
    uint32_t weightOffsetsTableOffset = std::numeric_limits<uint32_t>::max();
};

static_assert(sizeof(LinearPushConstants) == 40);
static_assert(offsetof(LinearPushConstants, transposeWeights) == 32);
static_assert(offsetof(LinearPushConstants, weightOffsetsTableOffset) == 36);

struct ResidentLinearPushConstants {
    uint32_t inputBatchStrideWords;
    uint32_t batchCount;
    uint32_t rowsPerBatch;
    uint32_t inputColumns;
    uint32_t outputColumns;
    uint32_t inputOffsetWords;
    uint32_t weightOffsetHalf;
    uint32_t outputOffsetWords;
    uint32_t transposeWeights;
    uint32_t weightOffsetsTableOffsetWords;
};

static_assert(sizeof(ResidentLinearPushConstants) == 40);
static_assert(offsetof(ResidentLinearPushConstants, batchCount) == 4);
static_assert(offsetof(ResidentLinearPushConstants, weightOffsetsTableOffsetWords) == 36);

struct ResidentAttentionPushConstants {
    uint32_t operation;
    uint32_t sequenceLength;
    uint32_t modelWidth;
    uint32_t headCount;
    uint32_t headDimension;
    uint32_t qkvOffsetWords;
    uint32_t scoreOffsetWords;
    uint32_t attentionOutputOffsetWords;
    uint32_t projectedOutputOffsetWords;
    uint32_t residualOffsetWords;
    uint32_t outputWeightOffsetHalf;
    uint32_t normalizedOutputOffsetWords;
};

static_assert(sizeof(ResidentAttentionPushConstants) == 48);
static_assert(offsetof(ResidentAttentionPushConstants, qkvOffsetWords) == 20);
static_assert(offsetof(ResidentAttentionPushConstants, outputWeightOffsetHalf) == 40);
static_assert(offsetof(ResidentAttentionPushConstants, normalizedOutputOffsetWords) == 44);

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
    bool hasDeviceIdentityProperties = false;
    std::array<uint8_t, VK_UUID_SIZE> deviceUuid{};
    std::array<uint8_t, VK_UUID_SIZE> driverUuid{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    bool memoryPropertiesAvailable = false;
    uint32_t hostVisibleMemoryTypeCount = 0;
    uint32_t hostVisibleDeviceLocalMemoryTypeCount = 0;
    bool memoryBudgetExtensionSupported = false;
    bool memoryBudgetExtensionEnabled = false;
    bool memoryBudgetQueryAvailable = false;
    uint32_t memoryBudgetHeapCount = 0;
    std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> memoryHeapBudget{};
    std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> memoryHeapUsage{};
#if defined(VK_VERSION_1_1)
    PFN_vkGetPhysicalDeviceMemoryProperties2 getPhysicalDeviceMemoryProperties2 = nullptr;
#endif
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    uint32_t computeQueueTimestampValidBits = 0;
    uint32_t effectiveApiVersion = VK_API_VERSION_1_0;
    uint32_t maxComputeWorkGroupCountX = 0;
    bool synchronization2ExtensionSupported = false;
    bool synchronization2Supported = false;
    bool synchronization2Enabled = false;
    PFN_vkQueueSubmit2KHR queueSubmit2 = nullptr;
    PFN_vkCmdPipelineBarrier2KHR cmdPipelineBarrier2 = nullptr;
    PFN_vkCmdWriteTimestamp2KHR cmdWriteTimestamp2 = nullptr;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence submissionFence = VK_NULL_HANDLE;
    VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
    bool timestampProfilingAvailable = false;
    uint64_t timestampedDispatchCount = 0;
    uint64_t lastAdamwGpuTimeNanoseconds = 0;
    uint64_t lastLinearGpuTimeNanoseconds = 0;
    uint64_t lastFp16LinearGpuTimeNanoseconds = 0;
    uint64_t totalGpuTimeNanoseconds = 0;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet tensorArenaDescriptorSet = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipeline linearPipeline = VK_NULL_HANDLE;
    VkPipeline fp16LinearPipeline = VK_NULL_HANDLE;
    VkPipeline residentLinearPipeline = VK_NULL_HANDLE;
    VkPipeline residentAttentionPipeline = VK_NULL_HANDLE;

    uint32_t instanceApiVersion = VK_API_VERSION_1_0;
    bool supportsFp16Compute = false;
    bool supportsShaderFloat16Feature = false;
    bool supportsShaderInt8 = false;
    bool supportsStorageBuffer16BitAccess = false;
    bool supportsStorageBuffer8BitAccess = false;
    bool shaderFloat16FeatureEnabled = false;
    bool shaderInt8FeatureEnabled = false;
    bool storageBuffer16BitAccessEnabled = false;
    bool storageBuffer8BitAccessEnabled = false;

    uint32_t subgroupSize = 0;
    uint32_t subgroupSupportedStages = 0;
    uint32_t subgroupSupportedOperations = 0;
    bool subgroupQuadOperationsInAllStages = false;
    bool supportsSubgroupSizeControl = false;
    bool subgroupSizeControlEnabled = false;
    bool supportsComputeFullSubgroups = false;
    uint32_t minSubgroupSize = 0;
    uint32_t maxSubgroupSize = 0;
    uint32_t maxComputeWorkgroupSubgroups = 0;
    uint32_t requiredSubgroupSizeStages = 0;

    bool supportsShaderIntegerDotProduct = false;
    bool shaderIntegerDotProductEnabled = false;
    bool integerDotProduct8BitUnsignedAccelerated = false;
    bool integerDotProduct8BitSignedAccelerated = false;
    bool integerDotProduct8BitMixedSignednessAccelerated = false;
    bool integerDotProduct4x8BitPackedUnsignedAccelerated = false;
    bool integerDotProduct4x8BitPackedSignedAccelerated = false;
    bool integerDotProduct4x8BitPackedMixedSignednessAccelerated = false;

    bool hasCooperativeMatrixExtension = false;
    bool supportsCooperativeMatrix = false;
    bool cooperativeMatrixEnabled = false;
    bool supportsCooperativeMatrixRobustBufferAccess = false;
    uint32_t cooperativeMatrixSupportedStages = 0;
    std::vector<CooperativeMatrixCapability> cooperativeMatrixTypes;

    bool armTensorsExtensionSupported = false;
    bool armTensorsFeatureQueryAvailable = false;
    bool armTensorsFeatureSupported = false;
    bool armTensorNonPackedSupported = false;
    bool armTensorShaderAccessSupported = false;
    bool armTensorDynamicIndexingSupported = false;
    bool armTensorNonUniformIndexingSupported = false;
    bool armTensorUpdateAfterBindSupported = false;
    bool armTensorDeviceFeaturesEnabled = false;
    uint32_t shaderTensorSupportedStages = 0;
    uint32_t armTensorMaxDimensionCount = 0;
    uint64_t armTensorMaxElements = 0;
    uint64_t armTensorMaxPerDimensionElements = 0;
    int64_t armTensorMaxStride = 0;
    uint64_t armTensorMaxSizeBytes = 0;
    uint32_t armTensorMaxShaderAccessArrayLength = 0;
    uint32_t armTensorMaxShaderAccessSizeBytes = 0;
    uint32_t armTensorMaxDescriptorSetStorageTensors = 0;
    uint32_t armTensorMaxPerStageDescriptorSetStorageTensors = 0;
    bool armTensorNonUniformIndexingNative = false;

    // ARM shader-core values are telemetry only; no vendor shader feature is enabled.
    bool armShaderCorePropertiesExtensionSupported = false;
    bool armShaderCoreBuiltinsExtensionSupported = false;
    bool armShaderCoreBuiltinsFeatureQueryAvailable = false;
    bool armShaderCoreBuiltinsFeatureSupported = false;
    bool armShaderCoreBuiltinsFeatureEnabled = false;
    uint32_t armShaderCoreCount = 0;
    uint32_t armShaderWarpsPerCore = 0;
    uint32_t armShaderFmaRate = 0;
    uint32_t armShaderPixelRate = 0;
    uint32_t armShaderTexelRate = 0;

    VkBuffer storageBuffer = VK_NULL_HANDLE;
    VkDeviceMemory storageMemory = VK_NULL_HANDLE;
    VkDeviceSize storageCapacity = 0;
    void* mappedStorage = nullptr;
    bool hostCoherent = false;

    // Persistent model tensor storage; storageBuffer remains the mapped staging buffer.
    VkBuffer tensorArenaBuffer = VK_NULL_HANDLE;
    VkDeviceMemory tensorArenaMemory = VK_NULL_HANDLE;
    VkDeviceSize tensorArenaCapacity = 0;
    bool tensorArenaDeviceLocal = false;
    argus::vulkan::TensorArenaCore tensorArena;
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

std::string formatUuid(const uint8_t* uuid) {
    if (uuid == nullptr) {
        return {};
    }

    static constexpr char kHexDigits[] = "0123456789ABCDEF";
    std::string formatted;
    formatted.reserve(VK_UUID_SIZE * 2u);
    for (size_t index = 0u; index < VK_UUID_SIZE; ++index) {
        const uint8_t value = uuid[index];
        formatted.push_back(kHexDigits[(value >> 4u) & 0x0Fu]);
        formatted.push_back(kHexDigits[value & 0x0Fu]);
    }
    return formatted;
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

void destroyTensorArena();

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
        destroyTensorArena();
        destroyStorageBuffer();

        if (gRuntime.residentAttentionPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(gRuntime.device, gRuntime.residentAttentionPipeline, nullptr);
        }
        if (gRuntime.residentLinearPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(gRuntime.device, gRuntime.residentLinearPipeline, nullptr);
        }
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
        if (gRuntime.timestampQueryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(gRuntime.device, gRuntime.timestampQueryPool, nullptr);
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


void refreshMemoryBudget() {
#if defined(VK_EXT_memory_budget) && defined(VK_VERSION_1_1)
    if (!gRuntime.memoryBudgetQueryAvailable ||
        gRuntime.getPhysicalDeviceMemoryProperties2 == nullptr ||
        gRuntime.physicalDevice == VK_NULL_HANDLE) {
        return;
    }

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProperties{};
    budgetProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

    VkPhysicalDeviceMemoryProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    properties2.pNext = &budgetProperties;

    gRuntime.getPhysicalDeviceMemoryProperties2(
        gRuntime.physicalDevice,
        &properties2
    );

    gRuntime.memoryProperties = properties2.memoryProperties;
    gRuntime.memoryPropertiesAvailable =
        gRuntime.memoryProperties.memoryHeapCount > 0u &&
        gRuntime.memoryProperties.memoryTypeCount > 0u;
    gRuntime.memoryBudgetHeapCount = gRuntime.memoryProperties.memoryHeapCount;
    gRuntime.memoryHeapBudget = {};
    gRuntime.memoryHeapUsage = {};

    for (uint32_t heapIndex = 0;
         heapIndex < gRuntime.memoryBudgetHeapCount;
         ++heapIndex) {
        gRuntime.memoryHeapBudget[heapIndex] = budgetProperties.heapBudget[heapIndex];
        gRuntime.memoryHeapUsage[heapIndex] = budgetProperties.heapUsage[heapIndex];
    }
#endif
}

bool hasMemoryBudgetHeadroom(uint32_t memoryTypeIndex, VkDeviceSize requiredBytes) {
    if (!gRuntime.memoryBudgetQueryAvailable ||
        memoryTypeIndex >= gRuntime.memoryProperties.memoryTypeCount) {
        return true;
    }

    const uint32_t heapIndex =
        gRuntime.memoryProperties.memoryTypes[memoryTypeIndex].heapIndex;
    if (heapIndex >= gRuntime.memoryBudgetHeapCount ||
        heapIndex >= VK_MAX_MEMORY_HEAPS) {
        return true;
    }

    const VkDeviceSize budget = gRuntime.memoryHeapBudget[heapIndex];
    const VkDeviceSize usage = gRuntime.memoryHeapUsage[heapIndex];
    // A missing or inconsistent report is advisory-only. Let Vulkan's allocator
    // return the authoritative allocation result in that case.
    if (budget == 0u || usage > budget) {
        return true;
    }

    const VkDeviceSize remainingBudget = budget - usage;
    const VkDeviceSize minimumReserve = 1024u * 1024u;
    const VkDeviceSize proportionalReserve = budget / 10u;
    const VkDeviceSize reserve = std::min(
        budget / 4u,
        std::max(minimumReserve, proportionalReserve)
    );
    if (requiredBytes <= remainingBudget &&
        remainingBudget - requiredBytes >= reserve) {
        return true;
    }

    __android_log_print(
        ANDROID_LOG_WARN,
        kLogTag,
        "Vulkan allocation deferred: request=%llu, heap headroom=%llu, budget=%llu, usage=%llu.",
        static_cast<unsigned long long>(requiredBytes),
        static_cast<unsigned long long>(remainingBudget),
        static_cast<unsigned long long>(budget),
        static_cast<unsigned long long>(usage)
    );
    return false;
}

bool createStorageBuffer(VkDeviceSize requiredBytes) {
    if (gRuntime.device == VK_NULL_HANDLE ||
        !gRuntime.memoryPropertiesAvailable ||
        gRuntime.memoryProperties.memoryTypeCount == 0u ||
        requiredBytes == 0 ||
        requiredBytes > gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange) {
        return false;
    }

    destroyStorageBuffer();

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = requiredBytes;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
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
    refreshMemoryBudget();

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
    if (!hasMemoryBudgetHeadroom(selectedMemoryType, requirements.size)) {
        logError("Vulkan storage allocation exceeds the currently reported memory-heap budget.");
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

bool submitAndWait(const char* operation) {
    if (gRuntime.device == VK_NULL_HANDLE ||
        gRuntime.computeQueue == VK_NULL_HANDLE ||
        gRuntime.submissionFence == VK_NULL_HANDLE ||
        gRuntime.queueSubmit2 == nullptr) {
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

    VkCommandBufferSubmitInfoKHR commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR;
    commandBufferInfo.commandBuffer = gRuntime.commandBuffer;
    commandBufferInfo.deviceMask = 0u;

    VkSubmitInfo2KHR submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR;
    submitInfo.commandBufferInfoCount = 1u;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;

    result = gRuntime.queueSubmit2(
        gRuntime.computeQueue,
        1u,
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


bool alignArenaSize(VkDeviceSize value, VkDeviceSize alignment, VkDeviceSize& output) {
    if (alignment == 0u) return false;
    const VkDeviceSize remainder = value % alignment;
    const VkDeviceSize padding = remainder == 0u ? 0u : alignment - remainder;
    if (value > std::numeric_limits<VkDeviceSize>::max() - padding) return false;
    output = value + padding;
    return true;
}

void destroyTensorArena() {
    if (gRuntime.device != VK_NULL_HANDLE) {
        if (gRuntime.tensorArenaBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(gRuntime.device, gRuntime.tensorArenaBuffer, nullptr);
        }
        if (gRuntime.tensorArenaMemory != VK_NULL_HANDLE) {
            vkFreeMemory(gRuntime.device, gRuntime.tensorArenaMemory, nullptr);
        }
    }
    gRuntime.tensorArenaBuffer = VK_NULL_HANDLE;
    gRuntime.tensorArenaMemory = VK_NULL_HANDLE;
    gRuntime.tensorArenaCapacity = 0u;
    gRuntime.tensorArenaDeviceLocal = false;
    gRuntime.tensorArena.reset();
}

bool copyBufferSync(
    VkBuffer source,
    VkBuffer destination,
    VkDeviceSize sourceCapacity,
    VkDeviceSize destinationCapacity,
    VkDeviceSize sourceOffset,
    VkDeviceSize destinationOffset,
    VkDeviceSize size,
    bool sourceHostWritten,
    bool destinationHostReadable
) {
    if (source == VK_NULL_HANDLE || destination == VK_NULL_HANDLE ||
        gRuntime.commandBuffer == VK_NULL_HANDLE ||
        gRuntime.cmdPipelineBarrier2 == nullptr ||
        !argus::vulkan::TensorArenaCore::isValidCopyRange(
            sourceCapacity, sourceOffset, destinationCapacity, destinationOffset, size
        )) {
        logError("Vulkan buffer copy rejected: invalid handles, alignment, or buffer range.");
        return false;
    }
    VkResult result = vkResetCommandBuffer(gRuntime.commandBuffer, 0);
    if (result != VK_SUCCESS) return false;

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(gRuntime.commandBuffer, &begin);
    if (result != VK_SUCCESS) return false;

    VkBufferMemoryBarrier2KHR before[2]{};
    before[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
    before[0].srcStageMask =
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR |
        VK_PIPELINE_STAGE_2_COPY_BIT_KHR |
        (sourceHostWritten ? VK_PIPELINE_STAGE_2_HOST_BIT_KHR : 0u);
    before[0].srcAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR | VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR |
        (sourceHostWritten ? VK_ACCESS_2_HOST_WRITE_BIT_KHR : 0u);
    before[0].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    before[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT_KHR;
    before[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before[0].buffer = source;
    before[0].offset = sourceOffset;
    before[0].size = size;

    // The destination range may have been read or written by an earlier GPU dispatch.
    before[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
    before[1].srcStageMask =
        VK_PIPELINE_STAGE_2_HOST_BIT_KHR |
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR |
        VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    before[1].srcAccessMask =
        VK_ACCESS_2_HOST_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR | VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
    before[1].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    before[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
    before[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before[1].buffer = destination;
    before[1].offset = destinationOffset;
    before[1].size = size;

    VkDependencyInfoKHR beforeDependency{};
    beforeDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    beforeDependency.bufferMemoryBarrierCount = 2u;
    beforeDependency.pBufferMemoryBarriers = before;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &beforeDependency);

    VkBufferCopy region{};
    region.srcOffset = sourceOffset;
    region.dstOffset = destinationOffset;
    region.size = size;
    vkCmdCopyBuffer(gRuntime.commandBuffer, source, destination, 1, &region);

    VkBufferMemoryBarrier2KHR after{};
    after.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
    after.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    after.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
    after.dstStageMask = destinationHostReadable
        ? VK_PIPELINE_STAGE_2_HOST_BIT_KHR
        : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    after.dstAccessMask = destinationHostReadable
        ? VK_ACCESS_2_HOST_READ_BIT_KHR
        : (VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR);
    after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after.buffer = destination;
    after.offset = destinationOffset;
    after.size = size;

    VkDependencyInfoKHR afterDependency{};
    afterDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    afterDependency.bufferMemoryBarrierCount = 1u;
    afterDependency.pBufferMemoryBarriers = &after;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &afterDependency);

    result = vkEndCommandBuffer(gRuntime.commandBuffer);
    if (result != VK_SUCCESS) return false;
    return submitAndWait("Vulkan tensor-arena transfer failed.");
}

bool createArenaBacking(
    VkDeviceSize capacity,
    VkBuffer& buffer,
    VkDeviceMemory& memory,
    bool& deviceLocal
) {
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    deviceLocal = false;
    if (gRuntime.device == VK_NULL_HANDLE || !gRuntime.memoryPropertiesAvailable) return false;

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = capacity;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(gRuntime.device, &info, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        logError("Failed to create the Vulkan tensor-arena buffer.", result);
        return false;
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(gRuntime.device, buffer, &requirements);
    refreshMemoryBudget();

    std::vector<uint32_t> candidates;
    try {
        for (uint32_t index = 0u;
             index < gRuntime.memoryProperties.memoryTypeCount;
             ++index) {
            if ((requirements.memoryTypeBits & (1u << index)) != 0u) {
                candidates.push_back(index);
            }
        }
    } catch (const std::bad_alloc&) {
        vkDestroyBuffer(gRuntime.device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }
    // Vulkan exposes at most VK_MAX_MEMORY_TYPES (32) candidates. A small
    // stable insertion sort avoids comparator-template edge cases in older
    // Android NDK libc++ while keeping the preferred memory types first.
    const auto memoryTypeScore = [](uint32_t index) noexcept -> uint32_t {
        const VkMemoryPropertyFlags flags =
            gRuntime.memoryProperties.memoryTypes[index].propertyFlags;
        return
            ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0u ? 100u : 0u) +
            ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0u ? 20u : 0u) +
            ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u ? 5u : 0u);
    };
    for (size_t index = 1u; index < candidates.size(); ++index) {
        const uint32_t candidate = candidates[index];
        const uint32_t candidateScore = memoryTypeScore(candidate);
        size_t insertionIndex = index;
        while (insertionIndex > 0u &&
               candidateScore > memoryTypeScore(candidates[insertionIndex - 1u])) {
            candidates[insertionIndex] = candidates[insertionIndex - 1u];
            --insertionIndex;
        }
        candidates[insertionIndex] = candidate;
    }

    for (uint32_t index : candidates) {
        if (!hasMemoryBudgetHeadroom(index, requirements.size)) continue;
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = index;
        VkDeviceMemory candidate = VK_NULL_HANDLE;
        result = vkAllocateMemory(gRuntime.device, &allocation, nullptr, &candidate);
        if (result != VK_SUCCESS) continue;
        result = vkBindBufferMemory(gRuntime.device, buffer, candidate, 0u);
        if (result != VK_SUCCESS) {
            vkFreeMemory(gRuntime.device, candidate, nullptr);
            continue;
        }
        memory = candidate;
        deviceLocal =
            (gRuntime.memoryProperties.memoryTypes[index].propertyFlags &
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0u;
        return true;
    }

    logError("No eligible Vulkan memory type could back the tensor arena.");
    vkDestroyBuffer(gRuntime.device, buffer, nullptr);
    buffer = VK_NULL_HANDLE;
    return false;
}

void updateTensorArenaDescriptorSet(VkBuffer buffer, VkDeviceSize range) {
    if (gRuntime.device == VK_NULL_HANDLE ||
        gRuntime.tensorArenaDescriptorSet == VK_NULL_HANDLE ||
        buffer == VK_NULL_HANDLE ||
        range == 0u) {
        return;
    }

    VkDescriptorBufferInfo descriptorInfo{};
    descriptorInfo.buffer = buffer;
    descriptorInfo.offset = 0u;
    descriptorInfo.range = range;

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = gRuntime.tensorArenaDescriptorSet;
    descriptorWrite.dstBinding = 0u;
    descriptorWrite.dstArrayElement = 0u;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrite.descriptorCount = 1u;
    descriptorWrite.pBufferInfo = &descriptorInfo;

    vkUpdateDescriptorSets(gRuntime.device, 1u, &descriptorWrite, 0u, nullptr);
}

void updateTensorArenaDescriptorSet() {
    updateTensorArenaDescriptorSet(
        gRuntime.tensorArenaBuffer,
        gRuntime.tensorArenaCapacity
    );
}

bool growTensorArena(VkDeviceSize additionalBytes) {
    const VkDeviceSize alignment = std::max<VkDeviceSize>(
        4u, gRuntime.physicalDeviceProperties.limits.minStorageBufferOffsetAlignment
    );
    const VkDeviceSize oldCapacity = gRuntime.tensorArenaCapacity;
    if (additionalBytes == 0u ||
        oldCapacity > std::numeric_limits<VkDeviceSize>::max() - additionalBytes) {
        return false;
    }
    const VkDeviceSize maximumRange =
        gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange;
    const VkDeviceSize maximumCapacity = maximumRange - (maximumRange % alignment);
    VkDeviceSize minimumCapacity = 0u;
    if (!alignArenaSize(oldCapacity + additionalBytes, alignment, minimumCapacity) ||
        minimumCapacity > maximumCapacity) {
        return false;
    }
    VkDeviceSize targetCapacity = minimumCapacity;
    if (oldCapacity == 0u) {
        targetCapacity = std::max<VkDeviceSize>(targetCapacity, 1024u * 1024u);
    } else {
        const VkDeviceSize growth = std::max<VkDeviceSize>(oldCapacity / 2u, 1024u * 1024u);
        if (oldCapacity <= std::numeric_limits<VkDeviceSize>::max() - growth) {
            targetCapacity = std::max(targetCapacity, oldCapacity + growth);
        }
    }
    if (!alignArenaSize(targetCapacity, alignment, targetCapacity)) return false;
    targetCapacity = std::min(targetCapacity, maximumCapacity);
    if (targetCapacity < minimumCapacity || targetCapacity > maximumRange) {
        return false;
    }

    if (!gRuntime.tensorArena.prepareCapacityExtension()) {
        logError("Not enough host memory to prepare the Vulkan tensor-arena capacity update.");
        return false;
    }

    VkBuffer nextBuffer = VK_NULL_HANDLE;
    VkDeviceMemory nextMemory = VK_NULL_HANDLE;
    bool nextDeviceLocal = false;
    VkDeviceSize createdCapacity = targetCapacity;
    bool created = createArenaBacking(
        targetCapacity, nextBuffer, nextMemory, nextDeviceLocal
    );
    if (!created && targetCapacity != minimumCapacity) {
        createdCapacity = minimumCapacity;
        created = createArenaBacking(
            minimumCapacity, nextBuffer, nextMemory, nextDeviceLocal
        );
    }
    if (!created) return false;

    if (gRuntime.tensorArenaBuffer != VK_NULL_HANDLE) {
        const VkResult idle = vkDeviceWaitIdle(gRuntime.device);
        if (idle != VK_SUCCESS) {
            vkDestroyBuffer(gRuntime.device, nextBuffer, nullptr);
            vkFreeMemory(gRuntime.device, nextMemory, nullptr);
            logError("Vulkan device did not become idle before preserving tensor data.", idle);
            return false;
        }
        if (!copyBufferSync(
                gRuntime.tensorArenaBuffer, nextBuffer,
                oldCapacity, createdCapacity,
                0u, 0u, oldCapacity, false, false
            )) {
            vkDestroyBuffer(gRuntime.device, nextBuffer, nullptr);
            vkFreeMemory(gRuntime.device, nextMemory, nullptr);
            logError("Could not preserve live tensors while growing the Vulkan arena.");
            return false;
        }
    }

    if (!gRuntime.tensorArena.extendCapacity(oldCapacity, createdCapacity)) {
        vkDestroyBuffer(gRuntime.device, nextBuffer, nullptr);
        vkFreeMemory(gRuntime.device, nextMemory, nullptr);
        logError("Could not commit the tensor-arena free range after buffer growth.");
        return false;
    }

    const VkBuffer previousBuffer = gRuntime.tensorArenaBuffer;
    const VkDeviceMemory previousMemory = gRuntime.tensorArenaMemory;
    gRuntime.tensorArenaBuffer = nextBuffer;
    gRuntime.tensorArenaMemory = nextMemory;
    gRuntime.tensorArenaCapacity = createdCapacity;
    gRuntime.tensorArenaDeviceLocal = nextDeviceLocal;
    updateTensorArenaDescriptorSet();

    if (previousBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(gRuntime.device, previousBuffer, nullptr);
        vkFreeMemory(gRuntime.device, previousMemory, nullptr);
    }
    return true;
}

using TensorArenaAllocation = argus::vulkan::TensorArenaCore::Allocation;

struct ScopedTensorArenaAllocation final {
    explicit ScopedTensorArenaAllocation(uint64_t allocationToken) noexcept
        : token(allocationToken) {}

    ~ScopedTensorArenaAllocation() {
        if (token != 0u) gRuntime.tensorArena.release(token);
    }

    ScopedTensorArenaAllocation(const ScopedTensorArenaAllocation&) = delete;
    ScopedTensorArenaAllocation& operator=(const ScopedTensorArenaAllocation&) = delete;

    uint64_t token;
};

TensorArenaAllocation* findArenaAllocation(jlong rawHandle) {
    if (rawHandle == 0) return nullptr;
    return gRuntime.tensorArena.find(static_cast<uint64_t>(rawHandle));
}

jlong allocateArenaTensor(VkDeviceSize logicalSize) {
    if (logicalSize == 0u ||
        logicalSize > static_cast<VkDeviceSize>(std::numeric_limits<jsize>::max()) ||
        logicalSize > gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange) {
        return 0;
    }
    VkDeviceSize allocatedSize = 0u;
    if (!alignArenaSize(logicalSize, 4u, allocatedSize)) return 0;
    const VkDeviceSize offsetAlignment = std::max<VkDeviceSize>(
        4u, gRuntime.physicalDeviceProperties.limits.minStorageBufferOffsetAlignment
    );
    const VkDeviceSize maxRange =
        gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange;
    uint64_t token = 0U;
    if (gRuntime.tensorArena.tryAllocate(logicalSize, maxRange, offsetAlignment, token)) {
        return static_cast<jlong>(token);
    }
    if (!growTensorArena(allocatedSize) ||
        !gRuntime.tensorArena.tryAllocate(logicalSize, maxRange, offsetAlignment, token)) {
        return 0;
    }
    return static_cast<jlong>(token);
}

bool syncStagingMemory(bool invalidate) {
    if (gRuntime.hostCoherent) return true;
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = gRuntime.storageMemory;
    range.offset = 0u;
    range.size = VK_WHOLE_SIZE;
    const VkResult result = invalidate
        ? vkInvalidateMappedMemoryRanges(gRuntime.device, 1u, &range)
        : vkFlushMappedMemoryRanges(gRuntime.device, 1u, &range);
    if (result != VK_SUCCESS) {
        logError(invalidate ? "Failed to invalidate Vulkan tensor staging memory."
                            : "Failed to flush Vulkan tensor staging memory.", result);
        return false;
    }
    return true;
}

bool transferTensorBytes(
    JNIEnv* env,
    TensorArenaAllocation& allocation,
    jbyteArray array,
    bool upload
) {
    if (array == nullptr ||
        env->GetArrayLength(array) != static_cast<jsize>(allocation.logicalSize)) {
        return false;
    }
    const VkDeviceSize chunkLimit = std::min<VkDeviceSize>(
        gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange,
        8u * 1024u * 1024u
    ) & ~VkDeviceSize{3u};
    if (chunkLimit < 4u) return false;

    VkDeviceSize cursor = 0u;
    while (cursor < allocation.allocatedSize) {
        const VkDeviceSize chunk = std::min(allocation.allocatedSize - cursor, chunkLimit);
        const VkDeviceSize actual = cursor < allocation.logicalSize
            ? std::min(chunk, allocation.logicalSize - cursor)
            : 0u;
        if (!ensureStorageBuffer(chunk) || gRuntime.mappedStorage == nullptr) return false;

        auto* mapped = static_cast<jbyte*>(gRuntime.mappedStorage);
        if (upload) {
            std::memset(mapped, 0, static_cast<size_t>(chunk));
            if (actual > 0u) {
                env->GetByteArrayRegion(
                    array, static_cast<jsize>(cursor), static_cast<jsize>(actual), mapped
                );
                if (env->ExceptionCheck()) return false;
            }
            if (!syncStagingMemory(false) ||
                !copyBufferSync(
                    gRuntime.storageBuffer, gRuntime.tensorArenaBuffer,
                    gRuntime.storageCapacity, gRuntime.tensorArenaCapacity,
                    0u, allocation.offset + cursor, chunk, true, false
                )) return false;
        } else {
            if (!copyBufferSync(
                    gRuntime.tensorArenaBuffer, gRuntime.storageBuffer,
                    gRuntime.tensorArenaCapacity, gRuntime.storageCapacity,
                    allocation.offset + cursor, 0u, chunk, false, true
                ) || !syncStagingMemory(true)) return false;
            if (actual > 0u) {
                env->SetByteArrayRegion(
                    array, static_cast<jsize>(cursor), static_cast<jsize>(actual), mapped
                );
                if (env->ExceptionCheck()) return false;
            }
        }
        cursor += chunk;
    }
    return true;
}



bool beginGpuTimestampCapture(VkCommandBuffer commandBuffer) {
    if (!gRuntime.timestampProfilingAvailable ||
        gRuntime.timestampQueryPool == VK_NULL_HANDLE ||
        commandBuffer == VK_NULL_HANDLE) {
        return false;
    }

    vkCmdResetQueryPool(
        commandBuffer,
        gRuntime.timestampQueryPool,
        0,
        2
    );
    gRuntime.cmdWriteTimestamp2(
        commandBuffer,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
        gRuntime.timestampQueryPool,
        0
    );
    return true;
}

void endGpuTimestampCapture(VkCommandBuffer commandBuffer, bool captureStarted) {
    if (!captureStarted) {
        return;
    }

    gRuntime.cmdWriteTimestamp2(
        commandBuffer,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
        gRuntime.timestampQueryPool,
        1
    );
}

bool readGpuTimestampDurationNanoseconds(uint64_t& durationNanoseconds) {
    durationNanoseconds = 0;
    if (!gRuntime.timestampProfilingAvailable ||
        gRuntime.timestampQueryPool == VK_NULL_HANDLE ||
        gRuntime.computeQueueTimestampValidBits == 0u) {
        return false;
    }

    std::array<uint64_t, 2> timestamps{};
    const VkResult result = vkGetQueryPoolResults(
        gRuntime.device,
        gRuntime.timestampQueryPool,
        0,
        static_cast<uint32_t>(timestamps.size()),
        sizeof(timestamps),
        timestamps.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan GPU timestamp results are unavailable.", result);
        return false;
    }

    const uint32_t validBits = gRuntime.computeQueueTimestampValidBits;
    uint64_t tickDelta = timestamps[1] - timestamps[0];
    if (validBits < 64u) {
        const uint64_t validMask = (uint64_t{1} << validBits) - 1u;
        tickDelta &= validMask;
    }

    const double elapsedNanoseconds =
        static_cast<double>(tickDelta) *
        static_cast<double>(gRuntime.physicalDeviceProperties.limits.timestampPeriod);
    if (!std::isfinite(elapsedNanoseconds) ||
        elapsedNanoseconds < 0.0 ||
        elapsedNanoseconds > static_cast<double>(std::numeric_limits<uint64_t>::max())) {
        return false;
    }

    durationNanoseconds = static_cast<uint64_t>(elapsedNanoseconds);
    return true;
}

void recordGpuTiming(uint64_t durationNanoseconds, uint64_t& operationDuration) {
    operationDuration = durationNanoseconds;
    if (std::numeric_limits<uint64_t>::max() - gRuntime.totalGpuTimeNanoseconds >=
        durationNanoseconds) {
        gRuntime.totalGpuTimeNanoseconds += durationNanoseconds;
    } else {
        gRuntime.totalGpuTimeNanoseconds = std::numeric_limits<uint64_t>::max();
    }
    if (gRuntime.timestampedDispatchCount < std::numeric_limits<uint64_t>::max()) {
        ++gRuntime.timestampedDispatchCount;
    }
}

bool initializeRuntime() {
    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "Argus";
    applicationInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    applicationInfo.pEngineName = "Argus On-Device Trainer";
    applicationInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
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
#if defined(VK_VERSION_1_4)
    gRuntime.instanceApiVersion = std::min(loaderApiVersion, VK_API_VERSION_1_4);
#elif defined(VK_VERSION_1_3)
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
                gRuntime.computeQueueTimestampValidBits = family.timestampValidBits;
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
    if (gRuntime.memoryProperties.memoryHeapCount == 0u ||
        gRuntime.memoryProperties.memoryTypeCount == 0u) {
        logError("Selected Vulkan physical device exposed no usable memory heaps or memory types.");
        return false;
    }
    gRuntime.memoryPropertiesAvailable = true;
    for (uint32_t memoryTypeIndex = 0;
         memoryTypeIndex < gRuntime.memoryProperties.memoryTypeCount;
         ++memoryTypeIndex) {
        const VkMemoryPropertyFlags memoryFlags =
            gRuntime.memoryProperties.memoryTypes[memoryTypeIndex].propertyFlags;
        const bool hostVisible =
            (memoryFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0u;
        if (hostVisible) {
            ++gRuntime.hostVisibleMemoryTypeCount;
            if ((memoryFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0u) {
                ++gRuntime.hostVisibleDeviceLocalMemoryTypeCount;
            }
        }
    }
    __android_log_print(
        ANDROID_LOG_INFO,
        kLogTag,
        "Vulkan memory properties: %u heaps, %u types, %u host-visible, %u host-visible/device-local.",
        gRuntime.memoryProperties.memoryHeapCount,
        gRuntime.memoryProperties.memoryTypeCount,
        gRuntime.hostVisibleMemoryTypeCount,
        gRuntime.hostVisibleDeviceLocalMemoryTypeCount
    );

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
    const uint32_t effectiveApiVersion = std::min(
        gRuntime.instanceApiVersion,
        physicalApiVersion
    );
    gRuntime.effectiveApiVersion = effectiveApiVersion;
    const bool physicalSupportsVulkan11 = effectiveApiVersion >= VK_API_VERSION_1_1;
    const bool physicalSupportsVulkan12 = effectiveApiVersion >= VK_API_VERSION_1_2;
#if defined(VK_VERSION_1_3)
    const bool physicalSupportsVulkan13 = effectiveApiVersion >= VK_API_VERSION_1_3;
#else
    const bool physicalSupportsVulkan13 = false;
#endif

#if defined(VK_EXT_memory_budget) && defined(VK_VERSION_1_1)
    gRuntime.memoryBudgetExtensionSupported =
        effectiveApiVersion >= VK_API_VERSION_1_1 &&
        hasDeviceExtension(deviceExtensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
#else
    gRuntime.memoryBudgetExtensionSupported = false;
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

    gRuntime.synchronization2ExtensionSupported = hasDeviceExtension(
        deviceExtensions,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME
    );
    const bool querySynchronization2 =
        physicalSupportsVulkan13 ||
        (physicalSupportsVulkan11 && gRuntime.synchronization2ExtensionSupported);

#if defined(VK_KHR_cooperative_matrix)
    gRuntime.hasCooperativeMatrixExtension = hasDeviceExtension(
        deviceExtensions,
        VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
    );
#endif

    // This remains capability telemetry until Argus has an ARM tensor pipeline.
    gRuntime.armTensorsExtensionSupported =
        physicalSupportsVulkan13 &&
        hasDeviceExtension(deviceExtensions, "VK_ARM_tensors");
#if defined(VK_ARM_tensors)
    gRuntime.armTensorsFeatureQueryAvailable =
        gRuntime.armTensorsExtensionSupported;
#else
    gRuntime.armTensorsFeatureQueryAvailable = false;
#endif

#if defined(VK_ARM_shader_core_properties)
    gRuntime.armShaderCorePropertiesExtensionSupported = hasDeviceExtension(
        deviceExtensions,
        VK_ARM_SHADER_CORE_PROPERTIES_EXTENSION_NAME
    );
#endif

#if defined(VK_ARM_shader_core_builtins)
    gRuntime.armShaderCoreBuiltinsExtensionSupported = hasDeviceExtension(
        deviceExtensions,
        VK_ARM_SHADER_CORE_BUILTINS_EXTENSION_NAME
    );
    const bool queryArmShaderCoreBuiltins =
        gRuntime.armShaderCoreBuiltinsExtensionSupported;
#else
    const bool queryArmShaderCoreBuiltins = false;
#endif

    // Query extension and promoted-core features before creating the logical device.
    // Synchronization2 is required by the native submit, barrier, and timestamp path.
    // FP16 is enabled only when supported; other capabilities remain telemetry until
    // their matching shader kernels and logical-device feature chains are implemented.
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

#if defined(VK_VERSION_1_2)
    VkPhysicalDevice8BitStorageFeatures supportedStorage8{};
    if (physicalSupportsVulkan12) {
        supportedStorage8.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
        appendFeature(supportedStorage8);
    }
#endif

    VkPhysicalDeviceSynchronization2FeaturesKHR supportedSynchronization2{};
    if (querySynchronization2) {
        supportedSynchronization2.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
        appendFeature(supportedSynchronization2);
    }

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

#if defined(VK_ARM_tensors)
    VkPhysicalDeviceTensorFeaturesARM supportedArmTensorFeatures{};
    if (gRuntime.armTensorsFeatureQueryAvailable) {
        supportedArmTensorFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM;
        appendFeature(supportedArmTensorFeatures);
    }
#endif

#if defined(VK_ARM_shader_core_builtins)
    VkPhysicalDeviceShaderCoreBuiltinsFeaturesARM supportedArmShaderCoreBuiltins{};
    if (queryArmShaderCoreBuiltins) {
        supportedArmShaderCoreBuiltins.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_FEATURES_ARM;
        appendFeature(supportedArmShaderCoreBuiltins);
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
        gRuntime.synchronization2Supported =
            querySynchronization2 &&
            supportedSynchronization2.synchronization2 == VK_TRUE;
        gRuntime.supportsShaderFloat16Feature =
            queryShaderFloat16 && supportedShaderFloat16.shaderFloat16 == VK_TRUE;
        gRuntime.supportsFp16Compute = gRuntime.supportsShaderFloat16Feature;
        gRuntime.supportsShaderInt8 =
            queryShaderFloat16 && supportedShaderFloat16.shaderInt8 == VK_TRUE;
#if defined(VK_VERSION_1_1)
        gRuntime.supportsStorageBuffer16BitAccess =
            physicalSupportsVulkan11 &&
            supportedStorage16.storageBuffer16BitAccess == VK_TRUE;
#endif
#if defined(VK_VERSION_1_2)
        gRuntime.supportsStorageBuffer8BitAccess =
            physicalSupportsVulkan12 &&
            supportedStorage8.storageBuffer8BitAccess == VK_TRUE;
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
#if defined(VK_ARM_tensors)
        if (gRuntime.armTensorsFeatureQueryAvailable) {
            gRuntime.armTensorsFeatureSupported =
                supportedArmTensorFeatures.tensors == VK_TRUE;
            gRuntime.armTensorNonPackedSupported =
                supportedArmTensorFeatures.tensorNonPacked == VK_TRUE;
            gRuntime.armTensorShaderAccessSupported =
                supportedArmTensorFeatures.shaderTensorAccess == VK_TRUE;
            gRuntime.armTensorDynamicIndexingSupported =
                supportedArmTensorFeatures.shaderStorageTensorArrayDynamicIndexing == VK_TRUE;
            gRuntime.armTensorNonUniformIndexingSupported =
                supportedArmTensorFeatures.shaderStorageTensorArrayNonUniformIndexing == VK_TRUE;
            gRuntime.armTensorUpdateAfterBindSupported =
                supportedArmTensorFeatures.descriptorBindingStorageTensorUpdateAfterBind == VK_TRUE;
        }
#endif
        gRuntime.armShaderCoreBuiltinsFeatureQueryAvailable =
            queryArmShaderCoreBuiltins && features2Available;
#if defined(VK_ARM_shader_core_builtins)
        if (gRuntime.armShaderCoreBuiltinsFeatureQueryAvailable) {
            gRuntime.armShaderCoreBuiltinsFeatureSupported =
                supportedArmShaderCoreBuiltins.shaderCoreBuiltins == VK_TRUE;
        }
#endif
    } else {
        gRuntime.supportsFp16Compute = false;
        gRuntime.supportsShaderInt8 = false;
        gRuntime.supportsStorageBuffer8BitAccess = false;
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

#if defined(VK_VERSION_1_1)
        VkPhysicalDeviceIDProperties idProperties{};
        idProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        appendProperty(idProperties);
#endif

#if defined(VK_ARM_tensors)
        VkPhysicalDeviceTensorPropertiesARM armTensorProperties{};
        if (gRuntime.armTensorsFeatureQueryAvailable) {
            armTensorProperties.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_PROPERTIES_ARM;
            appendProperty(armTensorProperties);
        }
#endif

#if defined(VK_ARM_shader_core_properties)
        VkPhysicalDeviceShaderCorePropertiesARM armShaderCoreProperties{};
        if (gRuntime.armShaderCorePropertiesExtensionSupported) {
            armShaderCoreProperties.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_ARM;
            appendProperty(armShaderCoreProperties);
        }
#endif

#if defined(VK_ARM_shader_core_builtins)
        VkPhysicalDeviceShaderCoreBuiltinsPropertiesARM armShaderCoreBuiltinsProperties{};
        if (gRuntime.armShaderCoreBuiltinsExtensionSupported) {
            armShaderCoreBuiltinsProperties.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_PROPERTIES_ARM;
            appendProperty(armShaderCoreBuiltinsProperties);
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

#if defined(VK_VERSION_1_1)
        std::copy_n(idProperties.deviceUUID, VK_UUID_SIZE, gRuntime.deviceUuid.begin());
        std::copy_n(idProperties.driverUUID, VK_UUID_SIZE, gRuntime.driverUuid.begin());
        gRuntime.hasDeviceIdentityProperties = true;
#endif

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
#if defined(VK_ARM_tensors)
        if (gRuntime.armTensorsFeatureQueryAvailable) {
            gRuntime.shaderTensorSupportedStages =
                armTensorProperties.shaderTensorSupportedStages;
            gRuntime.armTensorMaxDimensionCount =
                armTensorProperties.maxTensorDimensionCount;
            gRuntime.armTensorMaxElements = armTensorProperties.maxTensorElements;
            gRuntime.armTensorMaxPerDimensionElements =
                armTensorProperties.maxPerDimensionTensorElements;
            gRuntime.armTensorMaxStride = armTensorProperties.maxTensorStride;
            gRuntime.armTensorMaxSizeBytes = armTensorProperties.maxTensorSize;
            gRuntime.armTensorMaxShaderAccessArrayLength =
                armTensorProperties.maxTensorShaderAccessArrayLength;
            gRuntime.armTensorMaxShaderAccessSizeBytes =
                armTensorProperties.maxTensorShaderAccessSize;
            gRuntime.armTensorMaxDescriptorSetStorageTensors =
                armTensorProperties.maxDescriptorSetStorageTensors;
            gRuntime.armTensorMaxPerStageDescriptorSetStorageTensors =
                armTensorProperties.maxPerStageDescriptorSetStorageTensors;
            gRuntime.armTensorNonUniformIndexingNative =
                armTensorProperties.shaderStorageTensorArrayNonUniformIndexingNative == VK_TRUE;
        }
#endif
#if defined(VK_ARM_shader_core_properties)
        if (gRuntime.armShaderCorePropertiesExtensionSupported) {
            gRuntime.armShaderFmaRate = armShaderCoreProperties.fmaRate;
            gRuntime.armShaderPixelRate = armShaderCoreProperties.pixelRate;
            gRuntime.armShaderTexelRate = armShaderCoreProperties.texelRate;
        }
#endif
#if defined(VK_ARM_shader_core_builtins)
        if (gRuntime.armShaderCoreBuiltinsExtensionSupported) {
            gRuntime.armShaderCoreCount = armShaderCoreBuiltinsProperties.shaderCoreCount;
            gRuntime.armShaderWarpsPerCore = armShaderCoreBuiltinsProperties.shaderWarpsPerCore;
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

    if (!gRuntime.synchronization2Supported) {
        logError(
            "The Vulkan device does not support Synchronization2; using the Kotlin compute fallback."
        );
        return false;
    }

    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = gRuntime.queueFamilyIndex;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures2 enabledFeatures{};
    VkPhysicalDeviceShaderFloat16Int8Features enabledShaderFloat16{};
    VkPhysicalDeviceSynchronization2FeaturesKHR enabledSynchronization2{};
    std::vector<const char*> enabledDeviceExtensions;
#if defined(VK_EXT_memory_budget)
    if (gRuntime.memoryBudgetExtensionSupported) {
        enabledDeviceExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }
#endif
    if (gRuntime.effectiveApiVersion < VK_API_VERSION_1_3 &&
        gRuntime.synchronization2ExtensionSupported) {
        enabledDeviceExtensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    }

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount =
        static_cast<uint32_t>(enabledDeviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = enabledDeviceExtensions.empty()
        ? nullptr
        : enabledDeviceExtensions.data();
    enabledFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    enabledSynchronization2.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
    enabledSynchronization2.synchronization2 = VK_TRUE;
    void** enabledFeatureChainTail = &enabledFeatures.pNext;
    if (gRuntime.supportsFp16Compute) {
        enabledShaderFloat16.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        enabledShaderFloat16.shaderFloat16 = VK_TRUE;
        enabledShaderFloat16.pNext = nullptr;
        *enabledFeatureChainTail = &enabledShaderFloat16;
        enabledFeatureChainTail = &enabledShaderFloat16.pNext;
    }
    *enabledFeatureChainTail = &enabledSynchronization2;
    enabledSynchronization2.pNext = nullptr;
    deviceInfo.pNext = &enabledFeatures;
    deviceInfo.pEnabledFeatures = nullptr;

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

    gRuntime.shaderFloat16FeatureEnabled =
        enabledShaderFloat16.shaderFloat16 == VK_TRUE;
    gRuntime.shaderInt8FeatureEnabled =
        enabledShaderFloat16.shaderInt8 == VK_TRUE;
    gRuntime.synchronization2Enabled =
        enabledSynchronization2.synchronization2 == VK_TRUE;

    const bool useCoreSynchronization2 =
        gRuntime.effectiveApiVersion >= VK_API_VERSION_1_3;
    const char* queueSubmit2Name = useCoreSynchronization2 ? "vkQueueSubmit2" : "vkQueueSubmit2KHR";
    const char* pipelineBarrier2Name = useCoreSynchronization2 ? "vkCmdPipelineBarrier2" : "vkCmdPipelineBarrier2KHR";
    const char* writeTimestamp2Name = useCoreSynchronization2 ? "vkCmdWriteTimestamp2" : "vkCmdWriteTimestamp2KHR";
    gRuntime.queueSubmit2 = reinterpret_cast<PFN_vkQueueSubmit2KHR>(
        vkGetDeviceProcAddr(gRuntime.device, queueSubmit2Name)
    );
    gRuntime.cmdPipelineBarrier2 = reinterpret_cast<PFN_vkCmdPipelineBarrier2KHR>(
        vkGetDeviceProcAddr(gRuntime.device, pipelineBarrier2Name)
    );
    gRuntime.cmdWriteTimestamp2 = reinterpret_cast<PFN_vkCmdWriteTimestamp2KHR>(
        vkGetDeviceProcAddr(gRuntime.device, writeTimestamp2Name)
    );
    if (!gRuntime.synchronization2Enabled ||
        gRuntime.queueSubmit2 == nullptr ||
        gRuntime.cmdPipelineBarrier2 == nullptr ||
        gRuntime.cmdWriteTimestamp2 == nullptr) {
        logError("Vulkan Synchronization2 was enabled but one or more required device entry points are unavailable.");
        return false;
    }

#if defined(VK_EXT_memory_budget) && defined(VK_VERSION_1_1)
    gRuntime.memoryBudgetExtensionEnabled = gRuntime.memoryBudgetExtensionSupported;
    if (gRuntime.memoryBudgetExtensionEnabled) {
        gRuntime.getPhysicalDeviceMemoryProperties2 =
            reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
                vkGetInstanceProcAddr(
                    gRuntime.instance,
                    "vkGetPhysicalDeviceMemoryProperties2"
                )
            );
        gRuntime.memoryBudgetQueryAvailable =
            gRuntime.getPhysicalDeviceMemoryProperties2 != nullptr;
        if (!gRuntime.memoryBudgetQueryAvailable) {
            logError(
                "VK_EXT_memory_budget was enabled but vkGetPhysicalDeviceMemoryProperties2 is unavailable."
            );
        }
    }
#endif

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

    if (gRuntime.computeQueueTimestampValidBits > 0u &&
        gRuntime.physicalDeviceProperties.limits.timestampPeriod > 0.0f) {
        VkQueryPoolCreateInfo timestampPoolInfo{};
        timestampPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        timestampPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        timestampPoolInfo.queryCount = 2;
        const VkResult timestampPoolResult = vkCreateQueryPool(
            gRuntime.device,
            &timestampPoolInfo,
            nullptr,
            &gRuntime.timestampQueryPool
        );
        if (timestampPoolResult == VK_SUCCESS) {
            gRuntime.timestampProfilingAvailable = true;
        } else {
            logError(
                "Vulkan timestamp query pool could not be created; GPU profiling is disabled.",
                timestampPoolResult
            );
        }
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
        std::max({
            sizeof(AdamwPushConstants),
            sizeof(LinearPushConstants),
            sizeof(ResidentAttentionPushConstants)
        })
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

    // This kernel uses packed uint loads and unpackHalf2x16. It therefore supports
    // FP16 model storage without requiring the optional shaderFloat16 device feature.
    VkShaderModuleCreateInfo residentShaderInfo{};
    residentShaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    residentShaderInfo.codeSize = kArgusResidentLinearShaderSpvSize;
    residentShaderInfo.pCode =
        reinterpret_cast<const uint32_t*>(kArgusResidentLinearShaderSpv);

    VkShaderModule residentShaderModule = VK_NULL_HANDLE;
    VkResult residentResult = vkCreateShaderModule(
        gRuntime.device,
        &residentShaderInfo,
        nullptr,
        &residentShaderModule
    );
    if (residentResult == VK_SUCCESS) {
        VkPipelineShaderStageCreateInfo residentShaderStage{};
        residentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        residentShaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        residentShaderStage.module = residentShaderModule;
        residentShaderStage.pName = "main";

        VkComputePipelineCreateInfo residentPipelineInfo{};
        residentPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        residentPipelineInfo.stage = residentShaderStage;
        residentPipelineInfo.layout = gRuntime.pipelineLayout;

        residentResult = vkCreateComputePipelines(
            gRuntime.device,
            VK_NULL_HANDLE,
            1u,
            &residentPipelineInfo,
            nullptr,
            &gRuntime.residentLinearPipeline
        );
    }
    if (residentShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(gRuntime.device, residentShaderModule, nullptr);
    }
    if (residentResult != VK_SUCCESS) {
        gRuntime.residentLinearPipeline = VK_NULL_HANDLE;
        logError(
            "Resident activation-linear pipeline is unavailable; existing Vulkan paths remain active.",
            residentResult
        );
    }

    VkShaderModuleCreateInfo attentionShaderInfo{};
    attentionShaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    attentionShaderInfo.codeSize = kArgusResidentAttentionShaderSpvSize;
    attentionShaderInfo.pCode =
        reinterpret_cast<const uint32_t*>(kArgusResidentAttentionShaderSpv);

    VkShaderModule attentionShaderModule = VK_NULL_HANDLE;
    VkResult attentionResult = vkCreateShaderModule(
        gRuntime.device, &attentionShaderInfo, nullptr, &attentionShaderModule
    );
    if (attentionResult == VK_SUCCESS) {
        VkPipelineShaderStageCreateInfo attentionShaderStage{};
        attentionShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        attentionShaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        attentionShaderStage.module = attentionShaderModule;
        attentionShaderStage.pName = "main";

        VkComputePipelineCreateInfo attentionPipelineInfo{};
        attentionPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        attentionPipelineInfo.stage = attentionShaderStage;
        attentionPipelineInfo.layout = gRuntime.pipelineLayout;

        attentionResult = vkCreateComputePipelines(
            gRuntime.device, VK_NULL_HANDLE, 1u, &attentionPipelineInfo,
            nullptr, &gRuntime.residentAttentionPipeline
        );
    }
    if (attentionShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(gRuntime.device, attentionShaderModule, nullptr);
    }
    if (attentionResult != VK_SUCCESS) {
        gRuntime.residentAttentionPipeline = VK_NULL_HANDLE;
        logError(
            "Resident causal-attention pipeline is unavailable; the verified Kotlin fallback remains active.",
            attentionResult
        );
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
    poolSize.descriptorCount = 2u;

    VkDescriptorPoolCreateInfo descriptorPoolInfo{};
    descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolInfo.maxSets = 2u;
    descriptorPoolInfo.poolSizeCount = 1u;
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

    const std::array<VkDescriptorSetLayout, 2> setLayouts{
        gRuntime.descriptorSetLayout,
        gRuntime.descriptorSetLayout
    };
    std::array<VkDescriptorSet, 2> descriptorSets{};

    VkDescriptorSetAllocateInfo descriptorSetInfo{};
    descriptorSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetInfo.descriptorPool = gRuntime.descriptorPool;
    descriptorSetInfo.descriptorSetCount = static_cast<uint32_t>(descriptorSets.size());
    descriptorSetInfo.pSetLayouts = setLayouts.data();

    result = vkAllocateDescriptorSets(
        gRuntime.device,
        &descriptorSetInfo,
        descriptorSets.data()
    );
    if (result != VK_SUCCESS) {
        logError("Vulkan descriptor-set allocation failed.", result);
        return false;
    }
    gRuntime.descriptorSet = descriptorSets[0];
    gRuntime.tensorArenaDescriptorSet = descriptorSets[1];

    __android_log_print(
        ANDROID_LOG_INFO,
        kLogTag,
        "Initialized Vulkan optimizer on %s (API %u.%u.%u).",
        gRuntime.physicalDeviceProperties.deviceName,
        VK_API_VERSION_MAJOR(gRuntime.physicalDeviceProperties.apiVersion),
        VK_API_VERSION_MINOR(gRuntime.physicalDeviceProperties.apiVersion),
        VK_API_VERSION_PATCH(gRuntime.physicalDeviceProperties.apiVersion)
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
    uint64_t adamwGpuTimeNanoseconds = 0u;
    uint64_t timestampedAdamwPages = 0u;
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
        const bool timestampCapture = beginGpuTimestampCapture(gRuntime.commandBuffer);

        VkMemoryBarrier2KHR hostToComputeBarrier{};
        hostToComputeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
        hostToComputeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT_KHR;
        hostToComputeBarrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT_KHR;
        hostToComputeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
        hostToComputeBarrier.dstAccessMask =
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
        VkDependencyInfoKHR hostToComputeDependency{};
        hostToComputeDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
        hostToComputeDependency.memoryBarrierCount = 1u;
        hostToComputeDependency.pMemoryBarriers = &hostToComputeBarrier;
        gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &hostToComputeDependency);

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

        VkMemoryBarrier2KHR computeToHostBarrier{};
        computeToHostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
        computeToHostBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
        computeToHostBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
        computeToHostBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT_KHR;
        computeToHostBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT_KHR;
        VkDependencyInfoKHR computeToHostDependency{};
        computeToHostDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
        computeToHostDependency.memoryBarrierCount = 1u;
        computeToHostDependency.pMemoryBarriers = &computeToHostBarrier;
        gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &computeToHostDependency);

        endGpuTimestampCapture(gRuntime.commandBuffer, timestampCapture);

        result = vkEndCommandBuffer(gRuntime.commandBuffer);
        if (result != VK_SUCCESS) {
            logError("Failed to finish the Vulkan AdamW page command buffer.", result);
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }

        // The mapped page is read back before reuse, so completion remains synchronous.
        // A fence targets this dispatch instead of idling the entire queue.
        if (!submitAndWait("Vulkan AdamW page submission or fence wait failed.")) {
            releaseGroupReferences(env, groups);
            return JNI_FALSE;
        }
        if (timestampCapture) {
            uint64_t pageDurationNanoseconds = 0u;
            if (readGpuTimestampDurationNanoseconds(pageDurationNanoseconds)) {
                if (std::numeric_limits<uint64_t>::max() - adamwGpuTimeNanoseconds >=
                    pageDurationNanoseconds) {
                    adamwGpuTimeNanoseconds += pageDurationNanoseconds;
                } else {
                    adamwGpuTimeNanoseconds = std::numeric_limits<uint64_t>::max();
                }
                ++timestampedAdamwPages;
            }
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

    if (timestampedAdamwPages > 0u) {
        recordGpuTiming(
            adamwGpuTimeNanoseconds,
            gRuntime.lastAdamwGpuTimeNanoseconds
        );
        if (timestampedAdamwPages > 1u) {
            const uint64_t additionalSamples = timestampedAdamwPages - 1u;
            if (std::numeric_limits<uint64_t>::max() - gRuntime.timestampedDispatchCount >=
                additionalSamples) {
                gRuntime.timestampedDispatchCount += additionalSamples;
            } else {
                gRuntime.timestampedDispatchCount = std::numeric_limits<uint64_t>::max();
            }
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
    const bool timestampCapture = beginGpuTimestampCapture(gRuntime.commandBuffer);

    VkMemoryBarrier2KHR hostToComputeBarrier{};
    hostToComputeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
    hostToComputeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT_KHR;
    hostToComputeBarrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT_KHR;
    hostToComputeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    hostToComputeBarrier.dstAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    VkDependencyInfoKHR hostToComputeDependency{};
    hostToComputeDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    hostToComputeDependency.memoryBarrierCount = 1u;
    hostToComputeDependency.pMemoryBarriers = &hostToComputeBarrier;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &hostToComputeDependency);

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

    VkMemoryBarrier2KHR computeToHostBarrier{};
    computeToHostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
    computeToHostBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    computeToHostBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    computeToHostBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT_KHR;
    computeToHostBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT_KHR;
    VkDependencyInfoKHR computeToHostDependency{};
    computeToHostDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    computeToHostDependency.memoryBarrierCount = 1u;
    computeToHostDependency.pMemoryBarriers = &computeToHostBarrier;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &computeToHostDependency);

    endGpuTimestampCapture(gRuntime.commandBuffer, timestampCapture);

    result = vkEndCommandBuffer(gRuntime.commandBuffer);
    if (result != VK_SUCCESS) {
        logError("Failed to finish the Vulkan linear command buffer.", result);
        return JNI_FALSE;
    }

    if (!submitAndWait("Vulkan linear submission or fence wait failed.")) {
        return JNI_FALSE;
    }
    if (timestampCapture) {
        uint64_t durationNanoseconds = 0u;
        if (readGpuTimestampDurationNanoseconds(durationNanoseconds)) {
            recordGpuTiming(durationNanoseconds, gRuntime.lastLinearGpuTimeNanoseconds);
        }
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
    const bool timestampCapture = beginGpuTimestampCapture(gRuntime.commandBuffer);

    VkMemoryBarrier2KHR hostToComputeBarrier{};
    hostToComputeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
    hostToComputeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT_KHR;
    hostToComputeBarrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT_KHR;
    hostToComputeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    hostToComputeBarrier.dstAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    VkDependencyInfoKHR hostToComputeDependency{};
    hostToComputeDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    hostToComputeDependency.memoryBarrierCount = 1u;
    hostToComputeDependency.pMemoryBarriers = &hostToComputeBarrier;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &hostToComputeDependency);

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

    VkMemoryBarrier2KHR computeToHostBarrier{};
    computeToHostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
    computeToHostBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    computeToHostBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    computeToHostBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT_KHR;
    computeToHostBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT_KHR;
    VkDependencyInfoKHR computeToHostDependency{};
    computeToHostDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    computeToHostDependency.memoryBarrierCount = 1u;
    computeToHostDependency.pMemoryBarriers = &computeToHostBarrier;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &computeToHostDependency);

    endGpuTimestampCapture(gRuntime.commandBuffer, timestampCapture);

    result = vkEndCommandBuffer(gRuntime.commandBuffer);
    if (result != VK_SUCCESS) {
        logError("Failed to finish the Vulkan FP16 linear command buffer.", result);
        return JNI_FALSE;
    }

    if (!submitAndWait("Vulkan FP16 linear submission or fence wait failed.")) {
        return JNI_FALSE;
    }
    if (timestampCapture) {
        uint64_t durationNanoseconds = 0u;
        if (readGpuTimestampDurationNanoseconds(durationNanoseconds)) {
            recordGpuTiming(durationNanoseconds, gRuntime.lastFp16LinearGpuTimeNanoseconds);
        }
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

extern "C" JNIEXPORT jboolean JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeApplyFp16LinearWithArena(
    JNIEnv* env,
    jobject,
    jfloatArray inputArray,
    jlongArray weightHandleArray,
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
        gRuntime.tensorArenaDescriptorSet == VK_NULL_HANDLE ||
        gRuntime.tensorArenaBuffer == VK_NULL_HANDLE ||
        gRuntime.commandBuffer == VK_NULL_HANDLE ||
        gRuntime.queueSubmit2 == nullptr ||
        gRuntime.cmdPipelineBarrier2 == nullptr ||
        inputArray == nullptr ||
        weightHandleArray == nullptr ||
        outputArray == nullptr ||
        rowCount <= 0 ||
        batchCount <= 0 ||
        rowsPerBatch <= 0 ||
        static_cast<uint64_t>(batchCount) * static_cast<uint64_t>(rowsPerBatch) !=
            static_cast<uint64_t>(rowCount) ||
        inputColumns <= 0 ||
        outputColumns <= 0) {
        return JNI_FALSE;
    }

    const uint64_t rows = static_cast<uint64_t>(rowCount);
    const uint64_t batches = static_cast<uint64_t>(batchCount);
    const uint64_t rowsInBatch = static_cast<uint64_t>(rowsPerBatch);
    const uint64_t inputWidth = static_cast<uint64_t>(inputColumns);
    const uint64_t outputWidth = static_cast<uint64_t>(outputColumns);
    const uint64_t maximumArrayElements =
        static_cast<uint64_t>(std::numeric_limits<jsize>::max());

    if (rows > maximumArrayElements / inputWidth ||
        rows > maximumArrayElements / outputWidth ||
        inputWidth > maximumArrayElements / outputWidth) {
        return JNI_FALSE;
    }

    const uint64_t inputElements = rows * inputWidth;
    const uint64_t outputElements = rows * outputWidth;
    const uint64_t weightElements = inputWidth * outputWidth;
    if (inputElements > std::numeric_limits<uint32_t>::max() ||
        outputElements > std::numeric_limits<uint32_t>::max() ||
        weightElements > maximumArrayElements ||
        weightElements > std::numeric_limits<uint32_t>::max()) {
        return JNI_FALSE;
    }

    if (env->GetArrayLength(inputArray) != static_cast<jsize>(inputElements) ||
        env->GetArrayLength(outputArray) != static_cast<jsize>(outputElements) ||
        env->GetArrayLength(weightHandleArray) != batchCount ||
        env->ExceptionCheck()) {
        return JNI_FALSE;
    }

    std::vector<jlong> weightHandles;
    try {
        weightHandles.resize(static_cast<size_t>(batches));
    } catch (const std::bad_alloc&) {
        logError("Persistent FP16 batch handle staging allocation failed.");
        return JNI_FALSE;
    }
    env->GetLongArrayRegion(
        weightHandleArray,
        0,
        batchCount,
        weightHandles.data()
    );
    if (env->ExceptionCheck()) return JNI_FALSE;

    const uint64_t weightByteCount64 = weightElements * sizeof(uint16_t);
    const uint64_t inputByteCount64 = inputElements * sizeof(uint16_t);
    const uint64_t outputByteCount64 = outputElements * sizeof(float);
    if (weightByteCount64 == 0u ||
        inputByteCount64 == 0u ||
        outputByteCount64 == 0u ||
        weightByteCount64 > std::numeric_limits<VkDeviceSize>::max() ||
        inputByteCount64 > std::numeric_limits<VkDeviceSize>::max() ||
        outputByteCount64 > std::numeric_limits<VkDeviceSize>::max()) {
        return JNI_FALSE;
    }

    const VkDeviceSize weightByteCount = static_cast<VkDeviceSize>(weightByteCount64);
    const VkDeviceSize inputByteCount = static_cast<VkDeviceSize>(inputByteCount64);
    const VkDeviceSize outputByteCount = static_cast<VkDeviceSize>(outputByteCount64);

    VkDeviceSize inputPaddedBytes = 0u;
    if (!alignArenaSize(inputByteCount, 4u, inputPaddedBytes)) {
        return JNI_FALSE;
    }

    const VkDeviceSize tableByteCount =
        static_cast<VkDeviceSize>(batches * sizeof(uint32_t));
    if (inputPaddedBytes > std::numeric_limits<VkDeviceSize>::max() - outputByteCount ||
        inputPaddedBytes + outputByteCount >
            std::numeric_limits<VkDeviceSize>::max() - tableByteCount ||
        inputPaddedBytes > std::numeric_limits<VkDeviceSize>::max() - tableByteCount) {
        return JNI_FALSE;
    }
    const VkDeviceSize scratchByteCount =
        inputPaddedBytes + outputByteCount + tableByteCount;
    const VkDeviceSize inputAndTableByteCount =
        inputPaddedBytes + tableByteCount;

    const VkDeviceSize maximumStorageRange =
        gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange;
    if (weightByteCount > maximumStorageRange ||
        scratchByteCount > maximumStorageRange ||
        inputAndTableByteCount > maximumStorageRange ||
        outputByteCount > maximumStorageRange) {
        return JNI_FALSE;
    }

    constexpr uint64_t kTileRows = 8u;
    constexpr uint64_t kTileColumns = 16u;
    const uint64_t tileRowsPerBatch = 1u + (rowsInBatch - 1u) / kTileRows;
    const uint64_t tileColumnsPerBatch = 1u + (outputWidth - 1u) / kTileColumns;
    if (tileRowsPerBatch > std::numeric_limits<uint64_t>::max() / tileColumnsPerBatch) {
        return JNI_FALSE;
    }
    const uint64_t tilesPerBatch = tileRowsPerBatch * tileColumnsPerBatch;
    if (batches > std::numeric_limits<uint64_t>::max() / tilesPerBatch) {
        return JNI_FALSE;
    }
    const uint64_t totalTileCount = batches * tilesPerBatch;
    const uint32_t workGroupCount = calculateDispatchGroupCount(totalTileCount);
    if (workGroupCount == 0u) {
        logError("Persistent batched FP16 linear dispatch has no available workgroups.");
        return JNI_FALSE;
    }

    const jlong scratchHandle = allocateArenaTensor(scratchByteCount);
    if (scratchHandle == 0) {
        logError("Persistent batched FP16 linear scratch allocation failed.");
        return JNI_FALSE;
    }
    ScopedTensorArenaAllocation scratchRelease(static_cast<uint64_t>(scratchHandle));
    TensorArenaAllocation* scratchAllocation =
        findArenaAllocation(static_cast<uint64_t>(scratchHandle));
    if (scratchAllocation == nullptr ||
        scratchAllocation->logicalSize < scratchByteCount) {
        logError("Persistent batched FP16 linear scratch allocation is invalid.");
        return JNI_FALSE;
    }

    std::vector<TensorArenaAllocation*> weightAllocations;
    try {
        weightAllocations.reserve(static_cast<size_t>(batches));
    } catch (const std::bad_alloc&) {
        logError("Persistent FP16 batch allocation metadata staging failed.");
        return JNI_FALSE;
    }

    for (uint64_t batchIndex = 0u; batchIndex < batches; ++batchIndex) {
        const jlong handle = weightHandles[static_cast<size_t>(batchIndex)];
        if (handle <= 0) {
            logError("Persistent FP16 batch contains an invalid tensor handle.");
            return JNI_FALSE;
        }
        TensorArenaAllocation* allocation =
            findArenaAllocation(static_cast<uint64_t>(handle));
        if (allocation == nullptr ||
            allocation->logicalSize < weightByteCount ||
            (allocation->offset & 1u) != 0u ||
            allocation->offset / 2u > std::numeric_limits<uint32_t>::max() ||
            weightElements - 1u >
                std::numeric_limits<uint32_t>::max() - allocation->offset / 2u) {
            logError("Persistent FP16 batch references an invalid, undersized, or shader-unaddressable weight allocation.");
            return JNI_FALSE;
        }
        weightAllocations.push_back(allocation);
    }

    const VkDeviceSize scratchEnd = scratchAllocation->offset + scratchByteCount;
    const VkDeviceSize offsetTableByteOffset =
        scratchAllocation->offset + inputPaddedBytes + outputByteCount;
    if (scratchEnd < scratchAllocation->offset ||
        (scratchAllocation->offset & 3u) != 0u ||
        scratchAllocation->offset / 2u > std::numeric_limits<uint32_t>::max() ||
        (scratchAllocation->offset + inputPaddedBytes) / 4u >
            std::numeric_limits<uint32_t>::max() ||
        offsetTableByteOffset / 4u > std::numeric_limits<uint32_t>::max() ||
        offsetTableByteOffset / 4u + batches - 1u >
            std::numeric_limits<uint32_t>::max()) {
        logError("Persistent batched FP16 scratch offsets exceed shader address limits.");
        return JNI_FALSE;
    }

    const uint64_t maximumShaderIndex =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    const uint64_t inputOffsetHalf = scratchAllocation->offset / 2u;
    const uint64_t outputOffsetWords =
        (scratchAllocation->offset + inputPaddedBytes) / 4u;
    const uint64_t offsetTableWordIndex = offsetTableByteOffset / 4u;
    if (inputElements - 1u > maximumShaderIndex - inputOffsetHalf ||
        outputElements - 1u > maximumShaderIndex - outputOffsetWords ||
        batches - 1u > maximumShaderIndex - offsetTableWordIndex) {
        logError("Persistent batched FP16 tensor ranges exceed 32-bit shader indexing.");
        return JNI_FALSE;
    }

    const VkDeviceSize stagingCapacity = std::max(inputAndTableByteCount, outputByteCount);
    if (!ensureStorageBuffer(stagingCapacity) || gRuntime.mappedStorage == nullptr) {
        logError("Persistent batched FP16 staging-buffer allocation failed.");
        return JNI_FALSE;
    }

    auto* mappedWords = static_cast<uint32_t*>(gRuntime.mappedStorage);
    std::memset(mappedWords, 0, static_cast<size_t>(inputAndTableByteCount));

    constexpr size_t kInputConversionChunkElements = 8192u;
    std::array<jfloat, kInputConversionChunkElements> inputChunk{};
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
        if (env->ExceptionCheck()) return JNI_FALSE;

        for (size_t localIndex = 0u; localIndex < chunkCount; localIndex += 2u) {
            const uint64_t absoluteIndex = chunkOffset + localIndex;
            const float firstValue = inputChunk[localIndex];
            if (!std::isfinite(firstValue)) {
                logError("Persistent batched FP16 input contains a non-finite activation.");
                return JNI_FALSE;
            }
            const uint16_t lowHalf = floatToHalf(firstValue);
            if ((lowHalf & 0x7C00u) == 0x7C00u) {
                logError("Persistent batched FP16 input exceeds the finite binary16 range.");
                return JNI_FALSE;
            }

            uint32_t packed = static_cast<uint32_t>(lowHalf);
            if (localIndex + 1u < chunkCount) {
                const float secondValue = inputChunk[localIndex + 1u];
                if (!std::isfinite(secondValue)) {
                    logError("Persistent batched FP16 input contains a non-finite activation.");
                    return JNI_FALSE;
                }
                const uint16_t highHalf = floatToHalf(secondValue);
                if ((highHalf & 0x7C00u) == 0x7C00u) {
                    logError("Persistent batched FP16 input exceeds the finite binary16 range.");
                    return JNI_FALSE;
                }
                packed |= static_cast<uint32_t>(highHalf) << 16u;
            }
            mappedWords[absoluteIndex >> 1u] = packed;
        }
    }

    auto* offsetTableWords = mappedWords + inputPaddedBytes / sizeof(uint32_t);
    for (uint64_t batchIndex = 0u; batchIndex < batches; ++batchIndex) {
        const VkDeviceSize halfOffset =
            weightAllocations[static_cast<size_t>(batchIndex)]->offset / 2u;
        if (halfOffset > std::numeric_limits<uint32_t>::max()) return JNI_FALSE;
        offsetTableWords[batchIndex] = static_cast<uint32_t>(halfOffset);
    }

    if (!syncStagingMemory(false)) return JNI_FALSE;

    VkResult result = vkResetCommandBuffer(gRuntime.commandBuffer, 0u);
    if (result != VK_SUCCESS) {
        logError("Failed to reset the persistent batched FP16 command buffer.", result);
        return JNI_FALSE;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(gRuntime.commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        logError("Failed to begin the persistent batched FP16 command buffer.", result);
        return JNI_FALSE;
    }

    VkBufferMemoryBarrier2KHR beforeInputCopy[2]{};
    beforeInputCopy[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
    beforeInputCopy[0].srcStageMask =
        VK_PIPELINE_STAGE_2_HOST_BIT_KHR | VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    beforeInputCopy[0].srcAccessMask =
        VK_ACCESS_2_HOST_WRITE_BIT_KHR |
        VK_ACCESS_2_HOST_READ_BIT_KHR |
        VK_ACCESS_2_TRANSFER_READ_BIT_KHR |
        VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
    beforeInputCopy[0].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    beforeInputCopy[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT_KHR;
    beforeInputCopy[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeInputCopy[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeInputCopy[0].buffer = gRuntime.storageBuffer;
    beforeInputCopy[0].offset = 0u;
    beforeInputCopy[0].size = inputAndTableByteCount;

    beforeInputCopy[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
    beforeInputCopy[1].srcStageMask =
        VK_PIPELINE_STAGE_2_COPY_BIT_KHR |
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    beforeInputCopy[1].srcAccessMask =
        VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    beforeInputCopy[1].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    beforeInputCopy[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
    beforeInputCopy[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeInputCopy[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeInputCopy[1].buffer = gRuntime.tensorArenaBuffer;
    beforeInputCopy[1].offset = scratchAllocation->offset;
    beforeInputCopy[1].size = scratchByteCount;

    VkDependencyInfoKHR beforeInputDependency{};
    beforeInputDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    beforeInputDependency.bufferMemoryBarrierCount = 2u;
    beforeInputDependency.pBufferMemoryBarriers = beforeInputCopy;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &beforeInputDependency);

    VkBufferCopy inputAndTableCopies[2]{};
    inputAndTableCopies[0].srcOffset = 0u;
    inputAndTableCopies[0].dstOffset = scratchAllocation->offset;
    inputAndTableCopies[0].size = inputPaddedBytes;
    inputAndTableCopies[1].srcOffset = inputPaddedBytes;
    inputAndTableCopies[1].dstOffset = offsetTableByteOffset;
    inputAndTableCopies[1].size = tableByteCount;
    vkCmdCopyBuffer(
        gRuntime.commandBuffer,
        gRuntime.storageBuffer,
        gRuntime.tensorArenaBuffer,
        2u,
        inputAndTableCopies
    );

    VkMemoryBarrier2KHR beforeCompute{};
    beforeCompute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
    beforeCompute.srcStageMask =
        VK_PIPELINE_STAGE_2_COPY_BIT_KHR |
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    beforeCompute.srcAccessMask =
        VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    beforeCompute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    beforeCompute.dstAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    VkDependencyInfoKHR beforeComputeDependency{};
    beforeComputeDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    beforeComputeDependency.memoryBarrierCount = 1u;
    beforeComputeDependency.pMemoryBarriers = &beforeCompute;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &beforeComputeDependency);

    const bool timestampCapture = beginGpuTimestampCapture(gRuntime.commandBuffer);
    vkCmdBindPipeline(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.fp16LinearPipeline
    );
    vkCmdBindDescriptorSets(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.pipelineLayout,
        0u,
        1u,
        &gRuntime.tensorArenaDescriptorSet,
        0u,
        nullptr
    );

    const LinearPushConstants pushConstants{
        static_cast<uint32_t>(rowCount),
        static_cast<uint32_t>(batchCount),
        static_cast<uint32_t>(rowsPerBatch),
        static_cast<uint32_t>(inputColumns),
        static_cast<uint32_t>(outputColumns),
        static_cast<uint32_t>(inputOffsetHalf),
        0u,
        static_cast<uint32_t>(outputOffsetWords),
        transposeWeights == JNI_TRUE ? 1u : 0u,
        static_cast<uint32_t>(offsetTableWordIndex)
    };
    vkCmdPushConstants(
        gRuntime.commandBuffer,
        gRuntime.pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0u,
        sizeof(LinearPushConstants),
        &pushConstants
    );
    vkCmdDispatch(gRuntime.commandBuffer, workGroupCount, 1u, 1u);
    endGpuTimestampCapture(gRuntime.commandBuffer, timestampCapture);

    VkBufferMemoryBarrier2KHR outputToCopy{};
    outputToCopy.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
    outputToCopy.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    outputToCopy.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    outputToCopy.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    outputToCopy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT_KHR;
    outputToCopy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputToCopy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputToCopy.buffer = gRuntime.tensorArenaBuffer;
    outputToCopy.offset = scratchAllocation->offset + inputPaddedBytes;
    outputToCopy.size = outputByteCount;

    VkBufferMemoryBarrier2KHR stagingBeforeOutput{};
    stagingBeforeOutput.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
    stagingBeforeOutput.srcStageMask =
        VK_PIPELINE_STAGE_2_HOST_BIT_KHR |
        VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    stagingBeforeOutput.srcAccessMask =
        VK_ACCESS_2_HOST_READ_BIT_KHR |
        VK_ACCESS_2_HOST_WRITE_BIT_KHR |
        VK_ACCESS_2_TRANSFER_READ_BIT_KHR |
        VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
    stagingBeforeOutput.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    stagingBeforeOutput.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
    stagingBeforeOutput.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    stagingBeforeOutput.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    stagingBeforeOutput.buffer = gRuntime.storageBuffer;
    stagingBeforeOutput.offset = 0u;
    stagingBeforeOutput.size = outputByteCount;

    VkBufferMemoryBarrier2KHR outputBarriers[2]{};
    outputBarriers[0] = outputToCopy;
    outputBarriers[1] = stagingBeforeOutput;
    VkDependencyInfoKHR outputDependency{};
    outputDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    outputDependency.bufferMemoryBarrierCount = 2u;
    outputDependency.pBufferMemoryBarriers = outputBarriers;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &outputDependency);

    VkBufferCopy outputCopy{};
    outputCopy.srcOffset = scratchAllocation->offset + inputPaddedBytes;
    outputCopy.dstOffset = 0u;
    outputCopy.size = outputByteCount;
    vkCmdCopyBuffer(
        gRuntime.commandBuffer,
        gRuntime.tensorArenaBuffer,
        gRuntime.storageBuffer,
        1u,
        &outputCopy
    );

    VkBufferMemoryBarrier2KHR outputToHost{};
    outputToHost.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
    outputToHost.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    outputToHost.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
    outputToHost.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT_KHR;
    outputToHost.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT_KHR;
    outputToHost.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputToHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputToHost.buffer = gRuntime.storageBuffer;
    outputToHost.offset = 0u;
    outputToHost.size = outputByteCount;

    VkDependencyInfoKHR outputToHostDependency{};
    outputToHostDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    outputToHostDependency.bufferMemoryBarrierCount = 1u;
    outputToHostDependency.pBufferMemoryBarriers = &outputToHost;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &outputToHostDependency);

    result = vkEndCommandBuffer(gRuntime.commandBuffer);
    if (result != VK_SUCCESS) {
        logError("Failed to finish the persistent batched FP16 command buffer.", result);
        return JNI_FALSE;
    }
    if (!submitAndWait("Persistent batched FP16 submission or fence wait failed.")) {
        return JNI_FALSE;
    }
    if (timestampCapture) {
        uint64_t durationNanoseconds = 0u;
        if (readGpuTimestampDurationNanoseconds(durationNanoseconds)) {
            recordGpuTiming(durationNanoseconds, gRuntime.lastFp16LinearGpuTimeNanoseconds);
        }
    }
    if (!syncStagingMemory(true)) return JNI_FALSE;

    const auto* outputValues = static_cast<const float*>(gRuntime.mappedStorage);
    for (uint64_t index = 0u; index < outputElements; ++index) {
        if (!std::isfinite(outputValues[index])) {
            logError("Persistent batched FP16 linear kernel produced a non-finite result.");
            return JNI_FALSE;
        }
    }
    env->SetFloatArrayRegion(
        outputArray,
        0,
        static_cast<jsize>(outputElements),
        outputValues
    );
    if (env->ExceptionCheck()) return JNI_FALSE;
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

    refreshMemoryBudget();
    const VkPhysicalDeviceProperties& properties = gRuntime.physicalDeviceProperties;
    std::ostringstream json;
    json << "{"
         << "\"deviceName\":\"" << escapeJsonString(properties.deviceName) << "\","
         << "\"vendorId\":" << properties.vendorID << ","
         << "\"deviceId\":" << properties.deviceID << ","
         << "\"deviceType\":" << static_cast<uint32_t>(properties.deviceType) << ","
         << "\"memoryPropertiesAvailable\":"
         << (gRuntime.memoryPropertiesAvailable ? "true" : "false") << ","
         << "\"memoryHeapCount\":" << gRuntime.memoryProperties.memoryHeapCount << ","
         << "\"memoryTypeCount\":" << gRuntime.memoryProperties.memoryTypeCount << ","
         << "\"hostVisibleMemoryTypeCount\":" << gRuntime.hostVisibleMemoryTypeCount << ","
         << "\"hostVisibleDeviceLocalMemoryTypeCount\":"
         << gRuntime.hostVisibleDeviceLocalMemoryTypeCount << ","
         << "\"memoryBudgetExtensionSupported\":"
         << (gRuntime.memoryBudgetExtensionSupported ? "true" : "false") << ","
         << "\"memoryBudgetExtensionEnabled\":"
         << (gRuntime.memoryBudgetExtensionEnabled ? "true" : "false") << ","
         << "\"memoryBudgetQueryAvailable\":"
         << (gRuntime.memoryBudgetQueryAvailable ? "true" : "false") << ","
         << "\"memoryBudgetHeapCount\":" << gRuntime.memoryBudgetHeapCount << ","
         << "\"memoryHeapBudgetBytes\":[";
    for (uint32_t heapIndex = 0; heapIndex < gRuntime.memoryBudgetHeapCount; ++heapIndex) {
        if (heapIndex > 0u) json << ",";
        json << gRuntime.memoryHeapBudget[heapIndex];
    }
    json << "],\"memoryHeapUsageBytes\":[";
    for (uint32_t heapIndex = 0; heapIndex < gRuntime.memoryBudgetHeapCount; ++heapIndex) {
        if (heapIndex > 0u) json << ",";
        json << gRuntime.memoryHeapUsage[heapIndex];
    }
    json << "],"
         << "\"apiVersionMajor\":" << VK_API_VERSION_MAJOR(properties.apiVersion) << ","
         << "\"apiVersionMinor\":" << VK_API_VERSION_MINOR(properties.apiVersion) << ","
         << "\"apiVersionPatch\":" << VK_API_VERSION_PATCH(properties.apiVersion) << ","
         << "\"effectiveApiVersionMajor\":" << VK_API_VERSION_MAJOR(gRuntime.effectiveApiVersion) << ","
         << "\"effectiveApiVersionMinor\":" << VK_API_VERSION_MINOR(gRuntime.effectiveApiVersion) << ","
         << "\"effectiveApiVersionPatch\":" << VK_API_VERSION_PATCH(gRuntime.effectiveApiVersion) << ","
         << "\"synchronization2ExtensionSupported\":"
         << (gRuntime.synchronization2ExtensionSupported ? "true" : "false") << ","
         << "\"synchronization2Supported\":"
         << (gRuntime.synchronization2Supported ? "true" : "false") << ","
         << "\"synchronization2Enabled\":"
         << (gRuntime.synchronization2Enabled ? "true" : "false") << ","
         << "\"driverName\":\"" << escapeJsonString(
                gRuntime.hasDriverProperties ? gRuntime.driverProperties.driverName : ""
            ) << "\","
         << "\"driverInfo\":\"" << escapeJsonString(
                gRuntime.hasDriverProperties ? gRuntime.driverProperties.driverInfo : ""
            ) << "\","
         << "\"driverVersionRaw\":" << properties.driverVersion << ","
         << "\"deviceIdentityPropertiesAvailable\":"
         << (gRuntime.hasDeviceIdentityProperties ? "true" : "false") << ","
         << "\"deviceUUID\":\"" << formatUuid(
                gRuntime.hasDeviceIdentityProperties ? gRuntime.deviceUuid.data() : nullptr
            ) << "\","
         << "\"driverUUID\":\"" << formatUuid(
                gRuntime.hasDeviceIdentityProperties ? gRuntime.driverUuid.data() : nullptr
            ) << "\","
         << "\"pipelineCacheUUID\":\"" << formatUuid(properties.pipelineCacheUUID) << "\","
         << "\"armShaderCorePropertiesExtensionSupported\":"
         << (gRuntime.armShaderCorePropertiesExtensionSupported ? "true" : "false") << ","
         << "\"armShaderCoreBuiltinsExtensionSupported\":"
         << (gRuntime.armShaderCoreBuiltinsExtensionSupported ? "true" : "false") << ","
         << "\"armShaderCoreBuiltinsFeatureQueryAvailable\":"
         << (gRuntime.armShaderCoreBuiltinsFeatureQueryAvailable ? "true" : "false") << ","
         << "\"armShaderCoreBuiltinsFeatureSupported\":"
         << (gRuntime.armShaderCoreBuiltinsFeatureSupported ? "true" : "false") << ","
         << "\"armShaderCoreBuiltinsFeatureEnabled\":"
         << (gRuntime.armShaderCoreBuiltinsFeatureEnabled ? "true" : "false") << ","
         << "\"armShaderCoreCount\":" << gRuntime.armShaderCoreCount << ","
         << "\"armShaderWarpsPerCore\":" << gRuntime.armShaderWarpsPerCore << ","
         << "\"armShaderFmaRate\":" << gRuntime.armShaderFmaRate << ","
         << "\"armShaderPixelRate\":" << gRuntime.armShaderPixelRate << ","
         << "\"armShaderTexelRate\":" << gRuntime.armShaderTexelRate << ","
         << "\"armTensorsExtensionSupported\":"
         << (gRuntime.armTensorsExtensionSupported ? "true" : "false") << ","
         << "\"armTensorsFeatureQueryAvailable\":"
         << (gRuntime.armTensorsFeatureQueryAvailable ? "true" : "false") << ","
         << "\"armTensorsFeatureSupported\":"
         << (gRuntime.armTensorsFeatureSupported ? "true" : "false") << ","
         << "\"armTensorNonPackedSupported\":"
         << (gRuntime.armTensorNonPackedSupported ? "true" : "false") << ","
         << "\"armTensorShaderAccessSupported\":"
         << (gRuntime.armTensorShaderAccessSupported ? "true" : "false") << ","
         << "\"armTensorDynamicIndexingSupported\":"
         << (gRuntime.armTensorDynamicIndexingSupported ? "true" : "false") << ","
         << "\"armTensorNonUniformIndexingSupported\":"
         << (gRuntime.armTensorNonUniformIndexingSupported ? "true" : "false") << ","
         << "\"armTensorUpdateAfterBindSupported\":"
         << (gRuntime.armTensorUpdateAfterBindSupported ? "true" : "false") << ","
         << "\"armTensorDeviceFeaturesEnabled\":"
         << (gRuntime.armTensorDeviceFeaturesEnabled ? "true" : "false") << ","
         << "\"shaderTensorSupportedStages\":" << gRuntime.shaderTensorSupportedStages << ","
         << "\"armTensorMaxDimensionCount\":" << gRuntime.armTensorMaxDimensionCount << ","
         << "\"armTensorMaxElements\":" << gRuntime.armTensorMaxElements << ","
         << "\"armTensorMaxPerDimensionElements\":"
         << gRuntime.armTensorMaxPerDimensionElements << ","
         << "\"armTensorMaxStride\":" << gRuntime.armTensorMaxStride << ","
         << "\"armTensorMaxSizeBytes\":" << gRuntime.armTensorMaxSizeBytes << ","
         << "\"armTensorMaxShaderAccessArrayLength\":"
         << gRuntime.armTensorMaxShaderAccessArrayLength << ","
         << "\"armTensorMaxShaderAccessSizeBytes\":"
         << gRuntime.armTensorMaxShaderAccessSizeBytes << ","
         << "\"armTensorMaxDescriptorSetStorageTensors\":"
         << gRuntime.armTensorMaxDescriptorSetStorageTensors << ","
         << "\"armTensorMaxPerStageDescriptorSetStorageTensors\":"
         << gRuntime.armTensorMaxPerStageDescriptorSetStorageTensors << ","
         << "\"armTensorNonUniformIndexingNative\":"
         << (gRuntime.armTensorNonUniformIndexingNative ? "true" : "false") << ","
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
         << "\"timestampPeriodNanoseconds\":"
         << properties.limits.timestampPeriod << ","
         << "\"timestampComputeAndGraphics\":"
         << (properties.limits.timestampComputeAndGraphics == VK_TRUE ? "true" : "false") << ","
         << "\"computeQueueTimestampValidBits\":"
         << gRuntime.computeQueueTimestampValidBits << ","
         << "\"computeTimestampQueriesSupported\":"
         << (gRuntime.computeQueueTimestampValidBits > 0u ? "true" : "false") << ","
         << "\"subgroupSize\":" << gRuntime.subgroupSize << ","
         << "\"subgroupSupportedStages\":" << gRuntime.subgroupSupportedStages << ","
         << "\"subgroupSupportedOperations\":" << gRuntime.subgroupSupportedOperations << ","
         << "\"subgroupQuadOperationsInAllStages\":"
         << (gRuntime.subgroupQuadOperationsInAllStages ? "true" : "false") << ","
         << "\"subgroupSizeControl\":"
         << (gRuntime.supportsSubgroupSizeControl ? "true" : "false") << ","
         << "\"subgroupSizeControlEnabled\":"
         << (gRuntime.subgroupSizeControlEnabled ? "true" : "false") << ","
         << "\"computeFullSubgroups\":"
         << (gRuntime.supportsComputeFullSubgroups ? "true" : "false") << ","
         << "\"minSubgroupSize\":" << gRuntime.minSubgroupSize << ","
         << "\"maxSubgroupSize\":" << gRuntime.maxSubgroupSize << ","
         << "\"maxComputeWorkgroupSubgroups\":"
         << gRuntime.maxComputeWorkgroupSubgroups << ","
         << "\"requiredSubgroupSizeStages\":" << gRuntime.requiredSubgroupSizeStages << ","
         << "\"shaderFloat16\":" << (gRuntime.supportsFp16Compute ? "true" : "false") << ","
         << "\"shaderFloat16FeatureSupported\":"
         << (gRuntime.supportsShaderFloat16Feature ? "true" : "false") << ","
         << "\"shaderFloat16FeatureEnabled\":"
         << (gRuntime.shaderFloat16FeatureEnabled ? "true" : "false") << ","
         << "\"shaderInt8\":" << (gRuntime.supportsShaderInt8 ? "true" : "false") << ","
         << "\"shaderInt8FeatureEnabled\":"
         << (gRuntime.shaderInt8FeatureEnabled ? "true" : "false") << ","
         << "\"storageBuffer16BitAccess\":"
         << (gRuntime.supportsStorageBuffer16BitAccess ? "true" : "false") << ","
         << "\"storageBuffer16BitAccessEnabled\":"
         << (gRuntime.storageBuffer16BitAccessEnabled ? "true" : "false") << ","
         << "\"storageBuffer8BitAccess\":"
         << (gRuntime.supportsStorageBuffer8BitAccess ? "true" : "false") << ","
         << "\"storageBuffer8BitAccessEnabled\":"
         << (gRuntime.storageBuffer8BitAccessEnabled ? "true" : "false") << ","
         << "\"shaderIntegerDotProduct\":"
         << (gRuntime.supportsShaderIntegerDotProduct ? "true" : "false") << ","
         << "\"shaderIntegerDotProductEnabled\":"
         << (gRuntime.shaderIntegerDotProductEnabled ? "true" : "false") << ","
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
         << "\"cooperativeMatrixEnabled\":"
         << (gRuntime.cooperativeMatrixEnabled ? "true" : "false") << ","
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


extern "C" JNIEXPORT jstring JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeGetGpuPerformanceMetrics(
    JNIEnv* env,
    jobject
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    std::ostringstream json;
    json << "{"
         << "\"timestampProfilingAvailable\":"
         << (gRuntime.timestampProfilingAvailable ? "true" : "false") << ","
         << "\"timestampedDispatchCount\":" << gRuntime.timestampedDispatchCount << ","
         << "\"lastAdamwGpuTimeNanoseconds\":" << gRuntime.lastAdamwGpuTimeNanoseconds << ","
         << "\"lastLinearGpuTimeNanoseconds\":" << gRuntime.lastLinearGpuTimeNanoseconds << ","
         << "\"lastFp16LinearGpuTimeNanoseconds\":" << gRuntime.lastFp16LinearGpuTimeNanoseconds << ","
         << "\"totalGpuTimeNanoseconds\":" << gRuntime.totalGpuTimeNanoseconds
         << "}";
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


extern "C" JNIEXPORT jboolean JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeClearTensorArena(
    JNIEnv*, jobject
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    if (gRuntime.device == VK_NULL_HANDLE) return JNI_FALSE;

    if (gRuntime.tensorArenaBuffer == VK_NULL_HANDLE) {
        gRuntime.tensorArena.reset();
        return JNI_TRUE;
    }

    const VkResult idle = vkDeviceWaitIdle(gRuntime.device);
    if (idle != VK_SUCCESS) {
        logError("Vulkan device did not become idle before clearing the tensor arena.", idle);
        return JNI_FALSE;
    }

    // Persistent FP16 dispatch is suspended by the Kotlin backend before this call.
    // Rebind descriptor set 1 to the still-live staging buffer before destroying the arena.
    if (gRuntime.storageBuffer == VK_NULL_HANDLE || gRuntime.storageCapacity == 0u) {
        logError("Cannot clear the Vulkan tensor arena without a valid fallback storage buffer.");
        return JNI_FALSE;
    }
    const VkDeviceSize maxStorageBufferRange = static_cast<VkDeviceSize>(
        gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange
    );
    const VkDeviceSize fallbackRange = std::min(
        gRuntime.storageCapacity,
        maxStorageBufferRange
    );
    if (fallbackRange == 0u) return JNI_FALSE;

    updateTensorArenaDescriptorSet(gRuntime.storageBuffer, fallbackRange);
    destroyTensorArena();
    refreshMemoryBudget();
    return JNI_TRUE;
}


extern "C" JNIEXPORT jboolean JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeApplyResidentLinear(
    JNIEnv*,
    jobject,
    jlong inputHandle,
    jlong weightHandle,
    jlong outputHandle,
    jint rowCount,
    jint inputColumns,
    jint outputColumns,
    jboolean transposeWeights
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);

    if (gRuntime.device == VK_NULL_HANDLE ||
        gRuntime.tensorArenaBuffer == VK_NULL_HANDLE ||
        gRuntime.tensorArenaDescriptorSet == VK_NULL_HANDLE ||
        gRuntime.residentLinearPipeline == VK_NULL_HANDLE ||
        inputHandle <= 0 ||
        weightHandle <= 0 ||
        outputHandle <= 0 ||
        inputHandle == weightHandle ||
        inputHandle == outputHandle ||
        weightHandle == outputHandle ||
        rowCount <= 0 ||
        inputColumns <= 0 ||
        outputColumns <= 0 ||
        (transposeWeights != JNI_TRUE && transposeWeights != JNI_FALSE)) {
        return JNI_FALSE;
    }

    TensorArenaAllocation* inputAllocation = findArenaAllocation(inputHandle);
    TensorArenaAllocation* weightAllocation = findArenaAllocation(weightHandle);
    TensorArenaAllocation* outputAllocation = findArenaAllocation(outputHandle);
    if (inputAllocation == nullptr ||
        weightAllocation == nullptr ||
        outputAllocation == nullptr) {
        logError("Resident linear dispatch references an unknown Tensor Arena allocation.");
        return JNI_FALSE;
    }

    const uint64_t rows = static_cast<uint64_t>(rowCount);
    const uint64_t inputWidth = static_cast<uint64_t>(inputColumns);
    const uint64_t outputWidth = static_cast<uint64_t>(outputColumns);
    if (rows > std::numeric_limits<uint64_t>::max() / inputWidth ||
        rows > std::numeric_limits<uint64_t>::max() / outputWidth ||
        inputWidth > std::numeric_limits<uint64_t>::max() / outputWidth) {
        return JNI_FALSE;
    }

    const uint64_t inputElements = rows * inputWidth;
    const uint64_t outputElements = rows * outputWidth;
    const uint64_t weightElements = inputWidth * outputWidth;
    constexpr uint64_t kMaximumShaderIndex =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    if (inputElements == 0u ||
        outputElements == 0u ||
        weightElements == 0u ||
        inputElements > kMaximumShaderIndex ||
        outputElements > kMaximumShaderIndex ||
        weightElements > kMaximumShaderIndex) {
        logError("Resident linear tensor dimensions exceed 32-bit shader indexing.");
        return JNI_FALSE;
    }

    const VkDeviceSize inputByteCount =
        static_cast<VkDeviceSize>(inputElements * sizeof(float));
    const VkDeviceSize outputByteCount =
        static_cast<VkDeviceSize>(outputElements * sizeof(float));
    const VkDeviceSize weightByteCount =
        static_cast<VkDeviceSize>(weightElements * sizeof(uint16_t));

    if (inputAllocation->logicalSize < inputByteCount ||
        weightAllocation->logicalSize < weightByteCount ||
        outputAllocation->logicalSize < outputByteCount ||
        (inputAllocation->offset & 3u) != 0u ||
        (outputAllocation->offset & 3u) != 0u ||
        (weightAllocation->offset & 1u) != 0u ||
        inputAllocation->offset > gRuntime.tensorArenaCapacity ||
        weightAllocation->offset > gRuntime.tensorArenaCapacity ||
        outputAllocation->offset > gRuntime.tensorArenaCapacity ||
        inputByteCount > gRuntime.tensorArenaCapacity - inputAllocation->offset ||
        weightByteCount > gRuntime.tensorArenaCapacity - weightAllocation->offset ||
        outputByteCount > gRuntime.tensorArenaCapacity - outputAllocation->offset) {
        logError("Resident linear tensor allocation is undersized, misaligned, or outside the arena.");
        return JNI_FALSE;
    }

    const uint64_t inputOffsetWords = inputAllocation->offset / sizeof(uint32_t);
    const uint64_t weightOffsetHalf = weightAllocation->offset / sizeof(uint16_t);
    const uint64_t outputOffsetWords = outputAllocation->offset / sizeof(uint32_t);
    if (inputOffsetWords > kMaximumShaderIndex ||
        weightOffsetHalf > kMaximumShaderIndex ||
        outputOffsetWords > kMaximumShaderIndex ||
        inputElements - 1u > kMaximumShaderIndex - inputOffsetWords ||
        weightElements - 1u > kMaximumShaderIndex - weightOffsetHalf ||
        outputElements - 1u > kMaximumShaderIndex - outputOffsetWords) {
        logError("Resident linear tensor ranges exceed 32-bit shader addressability.");
        return JNI_FALSE;
    }

    constexpr uint64_t kTileRows = 8u;
    constexpr uint64_t kTileColumns = 16u;
    const uint64_t tileRows = 1u + (rows - 1u) / kTileRows;
    const uint64_t tileColumns = 1u + (outputWidth - 1u) / kTileColumns;
    if (tileRows > std::numeric_limits<uint64_t>::max() / tileColumns) {
        return JNI_FALSE;
    }
    const uint64_t totalTileCount = tileRows * tileColumns;
    const uint32_t workGroupCount = calculateDispatchGroupCount(totalTileCount);
    if (workGroupCount == 0u) {
        logError("Resident linear dispatch has no available workgroups.");
        return JNI_FALSE;
    }

    VkResult result = vkResetCommandBuffer(gRuntime.commandBuffer, 0u);
    if (result != VK_SUCCESS) {
        logError("Failed to reset resident linear command buffer.", result);
        return JNI_FALSE;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(gRuntime.commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        logError("Failed to begin resident linear command buffer.", result);
        return JNI_FALSE;
    }

    VkMemoryBarrier2KHR beforeCompute{};
    beforeCompute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
    beforeCompute.srcStageMask =
        VK_PIPELINE_STAGE_2_COPY_BIT_KHR |
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    beforeCompute.srcAccessMask =
        VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    beforeCompute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    beforeCompute.dstAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;

    VkDependencyInfoKHR beforeDependency{};
    beforeDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    beforeDependency.memoryBarrierCount = 1u;
    beforeDependency.pMemoryBarriers = &beforeCompute;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &beforeDependency);

    const bool timestampCapture = beginGpuTimestampCapture(gRuntime.commandBuffer);
    vkCmdBindPipeline(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.residentLinearPipeline
    );
    vkCmdBindDescriptorSets(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.pipelineLayout,
        0u,
        1u,
        &gRuntime.tensorArenaDescriptorSet,
        0u,
        nullptr
    );

    const ResidentLinearPushConstants pushConstants{
        0u,
        1u,
        static_cast<uint32_t>(rowCount),
        static_cast<uint32_t>(inputColumns),
        static_cast<uint32_t>(outputColumns),
        static_cast<uint32_t>(inputOffsetWords),
        static_cast<uint32_t>(weightOffsetHalf),
        static_cast<uint32_t>(outputOffsetWords),
        transposeWeights == JNI_TRUE ? 1u : 0u,
        std::numeric_limits<uint32_t>::max()
    };
    vkCmdPushConstants(
        gRuntime.commandBuffer,
        gRuntime.pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0u,
        sizeof(LinearPushConstants),
        &pushConstants
    );
    vkCmdDispatch(gRuntime.commandBuffer, workGroupCount, 1u, 1u);
    endGpuTimestampCapture(gRuntime.commandBuffer, timestampCapture);

    // Make the FP32 output available to the next resident compute dispatch.
    VkMemoryBarrier2KHR afterCompute{};
    afterCompute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
    afterCompute.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    afterCompute.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    afterCompute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    afterCompute.dstAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    VkDependencyInfoKHR afterDependency{};
    afterDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    afterDependency.memoryBarrierCount = 1u;
    afterDependency.pMemoryBarriers = &afterCompute;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &afterDependency);

    result = vkEndCommandBuffer(gRuntime.commandBuffer);
    if (result != VK_SUCCESS) {
        logError("Failed to finish resident linear command buffer.", result);
        return JNI_FALSE;
    }
    if (!submitAndWait("Resident Vulkan linear dispatch failed.")) {
        return JNI_FALSE;
    }

    uint64_t elapsedNanoseconds = 0u;
    if (readGpuTimestampDurationNanoseconds(elapsedNanoseconds)) {
        gRuntime.lastLinearGpuTimeNanoseconds = elapsedNanoseconds;
        gRuntime.totalGpuTimeNanoseconds += elapsedNanoseconds;
        ++gRuntime.timestampedDispatchCount;
    }
    return JNI_TRUE;
}


extern "C" JNIEXPORT jboolean JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeApplyResidentLinearBatch(
    JNIEnv* env,
    jobject,
    jlong inputHandle,
    jlongArray weightHandleArray,
    jlong outputHandle,
    jint rowsPerBatch,
    jint inputColumns,
    jint outputColumns,
    jboolean transposeWeights
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);

    if (env == nullptr ||
        gRuntime.device == VK_NULL_HANDLE ||
        gRuntime.tensorArenaBuffer == VK_NULL_HANDLE ||
        gRuntime.tensorArenaDescriptorSet == VK_NULL_HANDLE ||
        gRuntime.residentLinearPipeline == VK_NULL_HANDLE ||
        gRuntime.commandBuffer == VK_NULL_HANDLE ||
        gRuntime.queueSubmit2 == nullptr ||
        gRuntime.cmdPipelineBarrier2 == nullptr ||
        inputHandle <= 0 ||
        outputHandle <= 0 ||
        inputHandle == outputHandle ||
        weightHandleArray == nullptr ||
        rowsPerBatch <= 0 ||
        inputColumns <= 0 ||
        outputColumns <= 0 ||
        (transposeWeights != JNI_TRUE && transposeWeights != JNI_FALSE)) {
        return JNI_FALSE;
    }

    const jsize batchCountValue = env->GetArrayLength(weightHandleArray);
    if (env->ExceptionCheck() || batchCountValue < 2 || batchCountValue > 8) {
        return JNI_FALSE;
    }

    const uint64_t batches = static_cast<uint64_t>(batchCountValue);
    const uint64_t rows = static_cast<uint64_t>(rowsPerBatch);
    const uint64_t inputWidth = static_cast<uint64_t>(inputColumns);
    const uint64_t outputWidth = static_cast<uint64_t>(outputColumns);
    constexpr uint64_t kMaximumShaderIndex =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());

    if (rows > std::numeric_limits<uint64_t>::max() / inputWidth ||
        rows > std::numeric_limits<uint64_t>::max() / outputWidth ||
        inputWidth > std::numeric_limits<uint64_t>::max() / outputWidth ||
        batches > std::numeric_limits<uint64_t>::max() / rows ||
        batches * rows > std::numeric_limits<uint64_t>::max() / outputWidth) {
        return JNI_FALSE;
    }

    const uint64_t inputElements = rows * inputWidth;
    const uint64_t outputElements = batches * rows * outputWidth;
    const uint64_t weightElements = inputWidth * outputWidth;
    if (inputElements == 0u ||
        outputElements == 0u ||
        weightElements == 0u ||
        inputElements > kMaximumShaderIndex ||
        outputElements > kMaximumShaderIndex ||
        weightElements > kMaximumShaderIndex) {
        logError("Resident projection-set dimensions exceed 32-bit shader indexing.");
        return JNI_FALSE;
    }

    const VkDeviceSize inputByteCount =
        static_cast<VkDeviceSize>(inputElements * sizeof(float));
    const VkDeviceSize outputByteCount =
        static_cast<VkDeviceSize>(outputElements * sizeof(float));
    const VkDeviceSize weightByteCount =
        static_cast<VkDeviceSize>(weightElements * sizeof(uint16_t));
    const VkDeviceSize tableByteCount =
        static_cast<VkDeviceSize>(batches * sizeof(uint32_t));
    const VkDeviceSize maximumStorageRange =
        gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange;
    if (inputByteCount == 0u ||
        outputByteCount == 0u ||
        weightByteCount == 0u ||
        tableByteCount == 0u ||
        inputByteCount > maximumStorageRange ||
        outputByteCount > maximumStorageRange ||
        weightByteCount > maximumStorageRange ||
        tableByteCount > maximumStorageRange) {
        return JNI_FALSE;
    }

    TensorArenaAllocation* inputAllocation =
        findArenaAllocation(static_cast<uint64_t>(inputHandle));
    TensorArenaAllocation* outputAllocation =
        findArenaAllocation(static_cast<uint64_t>(outputHandle));
    if (inputAllocation == nullptr || outputAllocation == nullptr) {
        logError("Resident projection set references an unknown input or output allocation.");
        return JNI_FALSE;
    }

    std::vector<jlong> weightHandles;
    std::vector<TensorArenaAllocation*> weightAllocations;
    try {
        weightHandles.resize(batches);
        weightAllocations.reserve(batches);
    } catch (const std::bad_alloc&) {
        logError("Resident projection-set metadata allocation failed.");
        return JNI_FALSE;
    }
    env->GetLongArrayRegion(
        weightHandleArray,
        0,
        batchCountValue,
        weightHandles.data()
    );
    if (env->ExceptionCheck()) return JNI_FALSE;

    if (inputAllocation->logicalSize < inputByteCount ||
        outputAllocation->logicalSize < outputByteCount ||
        (inputAllocation->offset & 3u) != 0u ||
        (outputAllocation->offset & 3u) != 0u ||
        inputAllocation->offset > gRuntime.tensorArenaCapacity ||
        outputAllocation->offset > gRuntime.tensorArenaCapacity ||
        inputByteCount > gRuntime.tensorArenaCapacity - inputAllocation->offset ||
        outputByteCount > gRuntime.tensorArenaCapacity - outputAllocation->offset) {
        logError("Resident projection-set input/output allocation is undersized or outside the arena.");
        return JNI_FALSE;
    }

    for (uint64_t batchIndex = 0u; batchIndex < batches; ++batchIndex) {
        const jlong handle = weightHandles[static_cast<size_t>(batchIndex)];
        if (handle <= 0 ||
            handle == inputHandle ||
            handle == outputHandle) {
            logError("Resident projection set contains an invalid or aliased weight handle.");
            return JNI_FALSE;
        }
        for (uint64_t previous = 0u; previous < batchIndex; ++previous) {
            if (weightHandles[static_cast<size_t>(previous)] == handle) {
                logError("Resident projection set contains duplicate weight handles.");
                return JNI_FALSE;
            }
        }

        TensorArenaAllocation* allocation =
            findArenaAllocation(static_cast<uint64_t>(handle));
        if (allocation == nullptr ||
            allocation->logicalSize < weightByteCount ||
            (allocation->offset & 1u) != 0u ||
            allocation->offset > gRuntime.tensorArenaCapacity ||
            weightByteCount > gRuntime.tensorArenaCapacity - allocation->offset ||
            allocation->offset / sizeof(uint16_t) > kMaximumShaderIndex ||
            weightElements - 1u >
                kMaximumShaderIndex - allocation->offset / sizeof(uint16_t)) {
            logError("Resident projection-set weight allocation is invalid or shader-unaddressable.");
            return JNI_FALSE;
        }
        weightAllocations.push_back(allocation);
    }

    const jlong tableHandle = allocateArenaTensor(tableByteCount);
    if (tableHandle == 0) {
        logError("Resident projection-set weight-offset table allocation failed.");
        return JNI_FALSE;
    }
    ScopedTensorArenaAllocation tableRelease(static_cast<uint64_t>(tableHandle));
    TensorArenaAllocation* tableAllocation =
        findArenaAllocation(static_cast<uint64_t>(tableHandle));
    if (tableAllocation == nullptr ||
        tableAllocation->logicalSize < tableByteCount ||
        (tableAllocation->offset & 3u) != 0u ||
        tableAllocation->offset > gRuntime.tensorArenaCapacity ||
        tableByteCount > gRuntime.tensorArenaCapacity - tableAllocation->offset) {
        logError("Resident projection-set weight-offset table allocation is invalid.");
        return JNI_FALSE;
    }

    const uint64_t inputOffsetWords =
        inputAllocation->offset / sizeof(uint32_t);
    const uint64_t outputOffsetWords =
        outputAllocation->offset / sizeof(uint32_t);
    const uint64_t tableOffsetWords =
        tableAllocation->offset / sizeof(uint32_t);
    if (inputOffsetWords > kMaximumShaderIndex ||
        outputOffsetWords > kMaximumShaderIndex ||
        tableOffsetWords > kMaximumShaderIndex ||
        inputElements - 1u > kMaximumShaderIndex - inputOffsetWords ||
        outputElements - 1u > kMaximumShaderIndex - outputOffsetWords ||
        batches - 1u > kMaximumShaderIndex - tableOffsetWords) {
        logError("Resident projection-set offsets exceed 32-bit shader addressability.");
        return JNI_FALSE;
    }

    if (!ensureStorageBuffer(tableByteCount) ||
        gRuntime.mappedStorage == nullptr) {
        logError("Resident projection-set offset-table staging allocation failed.");
        return JNI_FALSE;
    }
    auto* offsetTable = static_cast<uint32_t*>(gRuntime.mappedStorage);
    for (uint64_t batchIndex = 0u; batchIndex < batches; ++batchIndex) {
        offsetTable[batchIndex] = static_cast<uint32_t>(
            weightAllocations[static_cast<size_t>(batchIndex)]->offset /
            sizeof(uint16_t)
        );
    }
    if (!syncStagingMemory(false)) return JNI_FALSE;

    VkResult result = vkResetCommandBuffer(gRuntime.commandBuffer, 0u);
    if (result != VK_SUCCESS) {
        logError("Failed to reset resident projection-set command buffer.", result);
        return JNI_FALSE;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(gRuntime.commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        logError("Failed to begin resident projection-set command buffer.", result);
        return JNI_FALSE;
    }

    VkBufferMemoryBarrier2KHR beforeCopy[2]{};
    beforeCopy[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
    beforeCopy[0].srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT_KHR;
    beforeCopy[0].srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT_KHR;
    beforeCopy[0].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    beforeCopy[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT_KHR;
    beforeCopy[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeCopy[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeCopy[0].buffer = gRuntime.storageBuffer;
    beforeCopy[0].offset = 0u;
    beforeCopy[0].size = tableByteCount;

    beforeCopy[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
    beforeCopy[1].srcStageMask =
        VK_PIPELINE_STAGE_2_COPY_BIT_KHR |
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    beforeCopy[1].srcAccessMask =
        VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    beforeCopy[1].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
    beforeCopy[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
    beforeCopy[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeCopy[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeCopy[1].buffer = gRuntime.tensorArenaBuffer;
    beforeCopy[1].offset = tableAllocation->offset;
    beforeCopy[1].size = tableByteCount;

    VkDependencyInfoKHR beforeCopyDependency{};
    beforeCopyDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    beforeCopyDependency.bufferMemoryBarrierCount = 2u;
    beforeCopyDependency.pBufferMemoryBarriers = beforeCopy;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &beforeCopyDependency);

    VkBufferCopy tableCopy{};
    tableCopy.srcOffset = 0u;
    tableCopy.dstOffset = tableAllocation->offset;
    tableCopy.size = tableByteCount;
    vkCmdCopyBuffer(
        gRuntime.commandBuffer,
        gRuntime.storageBuffer,
        gRuntime.tensorArenaBuffer,
        1u,
        &tableCopy
    );

    VkMemoryBarrier2KHR beforeCompute{};
    beforeCompute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
    beforeCompute.srcStageMask =
        VK_PIPELINE_STAGE_2_COPY_BIT_KHR |
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    beforeCompute.srcAccessMask =
        VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    beforeCompute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    beforeCompute.dstAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    VkDependencyInfoKHR beforeComputeDependency{};
    beforeComputeDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    beforeComputeDependency.memoryBarrierCount = 1u;
    beforeComputeDependency.pMemoryBarriers = &beforeCompute;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &beforeComputeDependency);

    constexpr uint64_t kTileRows = 8u;
    constexpr uint64_t kTileColumns = 16u;
    const uint64_t tileRowsPerBatch = 1u + (rows - 1u) / kTileRows;
    const uint64_t tileColumnsPerBatch = 1u + (outputWidth - 1u) / kTileColumns;
    if (tileRowsPerBatch > std::numeric_limits<uint64_t>::max() / tileColumnsPerBatch ||
        batches > std::numeric_limits<uint64_t>::max() /
            (tileRowsPerBatch * tileColumnsPerBatch)) {
        return JNI_FALSE;
    }
    const uint64_t totalTileCount =
        batches * tileRowsPerBatch * tileColumnsPerBatch;
    const uint32_t workGroupCount = calculateDispatchGroupCount(totalTileCount);
    if (workGroupCount == 0u) {
        logError("Resident projection-set dispatch has no available workgroups.");
        return JNI_FALSE;
    }

    const ResidentLinearPushConstants pushConstants{
        0u,
        static_cast<uint32_t>(batches),
        static_cast<uint32_t>(rowsPerBatch),
        static_cast<uint32_t>(inputColumns),
        static_cast<uint32_t>(outputColumns),
        static_cast<uint32_t>(inputOffsetWords),
        0u,
        static_cast<uint32_t>(outputOffsetWords),
        transposeWeights == JNI_TRUE ? 1u : 0u,
        static_cast<uint32_t>(tableOffsetWords)
    };

    const bool timestampCapture = beginGpuTimestampCapture(gRuntime.commandBuffer);
    vkCmdBindPipeline(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.residentLinearPipeline
    );
    vkCmdBindDescriptorSets(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.pipelineLayout,
        0u,
        1u,
        &gRuntime.tensorArenaDescriptorSet,
        0u,
        nullptr
    );
    vkCmdPushConstants(
        gRuntime.commandBuffer,
        gRuntime.pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0u,
        sizeof(ResidentLinearPushConstants),
        &pushConstants
    );
    vkCmdDispatch(gRuntime.commandBuffer, workGroupCount, 1u, 1u);
    endGpuTimestampCapture(gRuntime.commandBuffer, timestampCapture);

    VkMemoryBarrier2KHR afterCompute{};
    afterCompute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
    afterCompute.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    afterCompute.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    afterCompute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    afterCompute.dstAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    VkDependencyInfoKHR afterComputeDependency{};
    afterComputeDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    afterComputeDependency.memoryBarrierCount = 1u;
    afterComputeDependency.pMemoryBarriers = &afterCompute;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &afterComputeDependency);

    result = vkEndCommandBuffer(gRuntime.commandBuffer);
    if (result != VK_SUCCESS) {
        logError("Failed to finish resident projection-set command buffer.", result);
        return JNI_FALSE;
    }
    if (!submitAndWait("Resident projection-set queue submission or fence wait failed.")) {
        return JNI_FALSE;
    }

    uint64_t elapsedNanoseconds = 0u;
    if (readGpuTimestampDurationNanoseconds(elapsedNanoseconds)) {
        gRuntime.lastLinearGpuTimeNanoseconds = elapsedNanoseconds;
        gRuntime.totalGpuTimeNanoseconds += elapsedNanoseconds;
        ++gRuntime.timestampedDispatchCount;
    }
    return JNI_TRUE;
}




extern "C" JNIEXPORT jboolean JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeRunResidentAttention(
    JNIEnv*,
    jobject,
    jlong qkvHandle,
    jlong scoreHandle,
    jlong attentionOutputHandle,
    jlong projectedOutputHandle,
    jlong normalizedOutputHandle,
    jlong residualHandle,
    jlong outputWeightHandle,
    jint sequenceLength,
    jint modelWidth,
    jint headCount,
    jint headDimension
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);

    if (gRuntime.device == VK_NULL_HANDLE ||
        gRuntime.tensorArenaBuffer == VK_NULL_HANDLE ||
        gRuntime.tensorArenaDescriptorSet == VK_NULL_HANDLE ||
        gRuntime.residentAttentionPipeline == VK_NULL_HANDLE ||
        gRuntime.commandBuffer == VK_NULL_HANDLE ||
        gRuntime.queueSubmit2 == nullptr ||
        gRuntime.cmdPipelineBarrier2 == nullptr ||
        qkvHandle <= 0 || scoreHandle <= 0 ||
        attentionOutputHandle <= 0 || projectedOutputHandle <= 0 ||
        normalizedOutputHandle <= 0 || residualHandle <= 0 || outputWeightHandle <= 0 ||
        sequenceLength <= 0 || modelWidth <= 0 ||
        headCount <= 0 || headDimension <= 0 ||
        static_cast<uint64_t>(headCount) * static_cast<uint64_t>(headDimension) !=
            static_cast<uint64_t>(modelWidth)) {
        return JNI_FALSE;
    }

    const uint64_t sequence = static_cast<uint64_t>(sequenceLength);
    const uint64_t width = static_cast<uint64_t>(modelWidth);
    const uint64_t heads = static_cast<uint64_t>(headCount);
    const uint64_t headWidth = static_cast<uint64_t>(headDimension);
    constexpr uint64_t kMaximumShaderIndex =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());

    if (sequence > std::numeric_limits<uint64_t>::max() / width ||
        sequence > std::numeric_limits<uint64_t>::max() / sequence ||
        heads > std::numeric_limits<uint64_t>::max() / sequence ||
        heads * sequence > std::numeric_limits<uint64_t>::max() / sequence ||
        width > std::numeric_limits<uint64_t>::max() / width) {
        return JNI_FALSE;
    }

    const uint64_t sequenceWidthElements = sequence * width;
    if (sequenceWidthElements > std::numeric_limits<uint64_t>::max() / 3u) {
        return JNI_FALSE;
    }
    const uint64_t qkvElements = sequenceWidthElements * 3u;
    const uint64_t scoreElements = heads * sequence * sequence;
    const uint64_t activationElements = sequenceWidthElements;
    const uint64_t weightElements = width * width;
    if (qkvElements == 0u || scoreElements == 0u ||
        activationElements == 0u || weightElements == 0u ||
        qkvElements > kMaximumShaderIndex ||
        scoreElements > kMaximumShaderIndex ||
        activationElements > kMaximumShaderIndex ||
        weightElements > kMaximumShaderIndex) {
        logError("Resident attention dimensions exceed 32-bit shader indexing.");
        return JNI_FALSE;
    }

    const VkDeviceSize qkvByteCount = static_cast<VkDeviceSize>(qkvElements * sizeof(float));
    const VkDeviceSize scoreByteCount = static_cast<VkDeviceSize>(scoreElements * sizeof(float));
    const VkDeviceSize activationByteCount =
        static_cast<VkDeviceSize>(activationElements * sizeof(float));
    const VkDeviceSize weightByteCount =
        static_cast<VkDeviceSize>(weightElements * sizeof(uint16_t));
    const VkDeviceSize maximumStorageRange =
        gRuntime.physicalDeviceProperties.limits.maxStorageBufferRange;
    if (qkvByteCount > maximumStorageRange ||
        scoreByteCount > maximumStorageRange ||
        activationByteCount > maximumStorageRange ||
        weightByteCount > maximumStorageRange) {
        logError("Resident attention tensors exceed maxStorageBufferRange.");
        return JNI_FALSE;
    }

    const jlong handles[] = {
        qkvHandle, scoreHandle, attentionOutputHandle,
        projectedOutputHandle, normalizedOutputHandle, residualHandle, outputWeightHandle
    };
    for (size_t index = 0u; index < 7u; ++index) {
        for (size_t previous = 0u; previous < index; ++previous) {
            if (handles[index] == handles[previous]) {
                logError("Resident attention tensors must use distinct arena allocations.");
                return JNI_FALSE;
            }
        }
    }

    TensorArenaAllocation* qkvAllocation =
        findArenaAllocation(static_cast<uint64_t>(qkvHandle));
    TensorArenaAllocation* scoreAllocation =
        findArenaAllocation(static_cast<uint64_t>(scoreHandle));
    TensorArenaAllocation* attentionOutputAllocation =
        findArenaAllocation(static_cast<uint64_t>(attentionOutputHandle));
    TensorArenaAllocation* projectedOutputAllocation =
        findArenaAllocation(static_cast<uint64_t>(projectedOutputHandle));
    TensorArenaAllocation* normalizedOutputAllocation =
        findArenaAllocation(static_cast<uint64_t>(normalizedOutputHandle));
    TensorArenaAllocation* residualAllocation =
        findArenaAllocation(static_cast<uint64_t>(residualHandle));
    TensorArenaAllocation* outputWeightAllocation =
        findArenaAllocation(static_cast<uint64_t>(outputWeightHandle));

    if (qkvAllocation == nullptr || scoreAllocation == nullptr ||
        attentionOutputAllocation == nullptr || projectedOutputAllocation == nullptr ||
        normalizedOutputAllocation == nullptr || residualAllocation == nullptr ||
        outputWeightAllocation == nullptr) {
        logError("Resident attention references an unknown Tensor Arena allocation.");
        return JNI_FALSE;
    }

    if (qkvAllocation->logicalSize < qkvByteCount ||
        scoreAllocation->logicalSize < scoreByteCount ||
        attentionOutputAllocation->logicalSize < activationByteCount ||
        projectedOutputAllocation->logicalSize < activationByteCount ||
        normalizedOutputAllocation->logicalSize < activationByteCount ||
        residualAllocation->logicalSize < activationByteCount ||
        outputWeightAllocation->logicalSize < weightByteCount) {
        logError("Resident attention allocation sizes do not match requested tensor shapes.");
        return JNI_FALSE;
    }

    const TensorArenaAllocation* floatAllocations[] = {
        qkvAllocation, scoreAllocation, attentionOutputAllocation,
        projectedOutputAllocation, normalizedOutputAllocation, residualAllocation
    };
    const VkDeviceSize floatByteCounts[] = {
        qkvByteCount, scoreByteCount, activationByteCount,
        activationByteCount, activationByteCount, activationByteCount
    };
    for (size_t index = 0u; index < 6u; ++index) {
        const TensorArenaAllocation& allocation = *floatAllocations[index];
        if ((allocation.offset & 3u) != 0u ||
            allocation.offset > gRuntime.tensorArenaCapacity ||
            floatByteCounts[index] > gRuntime.tensorArenaCapacity - allocation.offset) {
            logError("Resident attention FP32 allocation is misaligned or outside Tensor Arena.");
            return JNI_FALSE;
        }
    }
    if ((outputWeightAllocation->offset & 1u) != 0u ||
        outputWeightAllocation->offset > gRuntime.tensorArenaCapacity ||
        weightByteCount > gRuntime.tensorArenaCapacity - outputWeightAllocation->offset) {
        logError("Resident attention FP16 output weights are misaligned or outside Tensor Arena.");
        return JNI_FALSE;
    }

    const uint64_t qkvOffsetWords = qkvAllocation->offset / sizeof(uint32_t);
    const uint64_t scoreOffsetWords = scoreAllocation->offset / sizeof(uint32_t);
    const uint64_t attentionOutputOffsetWords =
        attentionOutputAllocation->offset / sizeof(uint32_t);
    const uint64_t projectedOutputOffsetWords =
        projectedOutputAllocation->offset / sizeof(uint32_t);
    const uint64_t normalizedOutputOffsetWords =
        normalizedOutputAllocation->offset / sizeof(uint32_t);
    const uint64_t residualOffsetWords = residualAllocation->offset / sizeof(uint32_t);
    const uint64_t outputWeightOffsetHalf =
        outputWeightAllocation->offset / sizeof(uint16_t);
    const uint64_t maximumIndex =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    if (qkvOffsetWords > maximumIndex || scoreOffsetWords > maximumIndex ||
        attentionOutputOffsetWords > maximumIndex ||
        projectedOutputOffsetWords > maximumIndex ||
        normalizedOutputOffsetWords > maximumIndex ||
        residualOffsetWords > maximumIndex || outputWeightOffsetHalf > maximumIndex ||
        qkvElements - 1u > maximumIndex - qkvOffsetWords ||
        scoreElements - 1u > maximumIndex - scoreOffsetWords ||
        activationElements - 1u > maximumIndex - attentionOutputOffsetWords ||
        activationElements - 1u > maximumIndex - projectedOutputOffsetWords ||
        activationElements - 1u > maximumIndex - normalizedOutputOffsetWords ||
        activationElements - 1u > maximumIndex - residualOffsetWords ||
        weightElements - 1u > maximumIndex - outputWeightOffsetHalf) {
        logError("Resident attention tensor indices exceed 32-bit shader addressability.");
        return JNI_FALSE;
    }

    const uint32_t scoreGroups = calculateDispatchWorkGroupCount(scoreElements);
    const uint32_t softmaxGroups = calculateDispatchWorkGroupCount(heads * sequence);
    const uint32_t activationGroups = calculateDispatchWorkGroupCount(activationElements);
    const uint32_t layerNormGroups = static_cast<uint32_t>(sequence);
    if (scoreGroups == 0u || softmaxGroups == 0u || activationGroups == 0u ||
        layerNormGroups == 0u ||
        sequence > static_cast<uint64_t>(gRuntime.maxComputeWorkGroupCountX)) {
        logError("Resident attention graph has no available compute workgroups.");
        return JNI_FALSE;
    }

    VkResult result = vkResetCommandBuffer(gRuntime.commandBuffer, 0u);
    if (result != VK_SUCCESS) {
        logError("Failed to reset resident attention command buffer.", result);
        return JNI_FALSE;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(gRuntime.commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        logError("Failed to begin resident attention command buffer.", result);
        return JNI_FALSE;
    }

    VkMemoryBarrier2KHR beforeCompute{};
    beforeCompute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
    beforeCompute.srcStageMask =
        VK_PIPELINE_STAGE_2_COPY_BIT_KHR |
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    beforeCompute.srcAccessMask =
        VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    beforeCompute.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    beforeCompute.dstAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    VkDependencyInfoKHR beforeDependency{};
    beforeDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    beforeDependency.memoryBarrierCount = 1u;
    beforeDependency.pMemoryBarriers = &beforeCompute;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &beforeDependency);

    vkCmdBindPipeline(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.residentAttentionPipeline
    );
    vkCmdBindDescriptorSets(
        gRuntime.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gRuntime.pipelineLayout,
        0u,
        1u,
        &gRuntime.tensorArenaDescriptorSet,
        0u,
        nullptr
    );

    const bool timestampCapture = beginGpuTimestampCapture(gRuntime.commandBuffer);
    const uint32_t groupCounts[] = {
        scoreGroups, softmaxGroups, activationGroups, activationGroups, layerNormGroups
    };
    VkMemoryBarrier2KHR betweenStages{};
    betweenStages.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
    betweenStages.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    betweenStages.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    betweenStages.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
    betweenStages.dstAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
    VkDependencyInfoKHR betweenDependency{};
    betweenDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    betweenDependency.memoryBarrierCount = 1u;
    betweenDependency.pMemoryBarriers = &betweenStages;

    for (uint32_t operation = 0u; operation < 5u; ++operation) {
        const ResidentAttentionPushConstants parameters{
            operation,
            static_cast<uint32_t>(sequence),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(heads),
            static_cast<uint32_t>(headWidth),
            static_cast<uint32_t>(qkvOffsetWords),
            static_cast<uint32_t>(scoreOffsetWords),
            static_cast<uint32_t>(attentionOutputOffsetWords),
            static_cast<uint32_t>(projectedOutputOffsetWords),
            static_cast<uint32_t>(residualOffsetWords),
            static_cast<uint32_t>(outputWeightOffsetHalf),
            static_cast<uint32_t>(normalizedOutputOffsetWords)
        };
        vkCmdPushConstants(
            gRuntime.commandBuffer,
            gRuntime.pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0u,
            sizeof(ResidentAttentionPushConstants),
            &parameters
        );
        vkCmdDispatch(gRuntime.commandBuffer, groupCounts[operation], 1u, 1u);
        if (operation != 4u) {
            gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &betweenDependency);
        }
    }
    endGpuTimestampCapture(gRuntime.commandBuffer, timestampCapture);

    VkBufferMemoryBarrier2KHR outputBarriers[4]{};
    const TensorArenaAllocation* outputAllocations[] = {
        scoreAllocation, attentionOutputAllocation, projectedOutputAllocation,
        normalizedOutputAllocation
    };
    const VkDeviceSize outputSizes[] = {
        scoreByteCount, activationByteCount, activationByteCount, activationByteCount
    };
    for (size_t index = 0u; index < 4u; ++index) {
        outputBarriers[index].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR;
        outputBarriers[index].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
        outputBarriers[index].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR;
        outputBarriers[index].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT_KHR;
        outputBarriers[index].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT_KHR;
        outputBarriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        outputBarriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        outputBarriers[index].buffer = gRuntime.tensorArenaBuffer;
        outputBarriers[index].offset = outputAllocations[index]->offset;
        outputBarriers[index].size = outputSizes[index];
    }
    VkDependencyInfoKHR outputDependency{};
    outputDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    outputDependency.bufferMemoryBarrierCount = 4u;
    outputDependency.pBufferMemoryBarriers = outputBarriers;
    gRuntime.cmdPipelineBarrier2(gRuntime.commandBuffer, &outputDependency);

    result = vkEndCommandBuffer(gRuntime.commandBuffer);
    if (result != VK_SUCCESS) {
        logError("Failed to finish resident attention command buffer.", result);
        return JNI_FALSE;
    }
    if (!submitAndWait("Resident attention graph submission or fence wait failed.")) {
        return JNI_FALSE;
    }

    uint64_t elapsedNanoseconds = 0u;
    if (readGpuTimestampDurationNanoseconds(elapsedNanoseconds)) {
        gRuntime.lastLinearGpuTimeNanoseconds = elapsedNanoseconds;
        gRuntime.totalGpuTimeNanoseconds += elapsedNanoseconds;
        ++gRuntime.timestampedDispatchCount;
    }
    return JNI_TRUE;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeAllocateTensor(
    JNIEnv*, jobject, jlong byteCount
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    if (gRuntime.device == VK_NULL_HANDLE || byteCount <= 0) return 0;
    return allocateArenaTensor(static_cast<VkDeviceSize>(byteCount));
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeUploadTensor(
    JNIEnv* env, jobject, jlong tensorHandle, jbyteArray source
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    if (gRuntime.device == VK_NULL_HANDLE || gRuntime.tensorArenaBuffer == VK_NULL_HANDLE) {
        return JNI_FALSE;
    }
    TensorArenaAllocation* allocation = findArenaAllocation(tensorHandle);
    return allocation != nullptr && transferTensorBytes(env, *allocation, source, true)
        ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeDownloadTensor(
    JNIEnv* env, jobject, jlong tensorHandle, jbyteArray destination
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    if (gRuntime.device == VK_NULL_HANDLE || gRuntime.tensorArenaBuffer == VK_NULL_HANDLE) {
        return JNI_FALSE;
    }
    TensorArenaAllocation* allocation = findArenaAllocation(tensorHandle);
    return allocation != nullptr && transferTensorBytes(env, *allocation, destination, false)
        ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeReleaseTensor(
    JNIEnv*, jobject, jlong tensorHandle
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    return gRuntime.tensorArena.release(static_cast<uint64_t>(tensorHandle))
        ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_argus_agi_model_VulkanTrainingBackend_nativeGetTensorArenaMetrics(
    JNIEnv* env, jobject
) {
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    std::ostringstream json;
    json << "{"
         << "\"capacityBytes\":" << gRuntime.tensorArena.capacityBytes() << ","
         << "\"allocatedBytes\":" << gRuntime.tensorArena.allocatedBytes() << ","
         << "\"activeAllocationCount\":" << gRuntime.tensorArena.activeAllocationCount() << ","
         << "\"deviceLocalMemory\":"
         << (gRuntime.tensorArenaDeviceLocal ? "true" : "false")
         << "}";
    const std::string payload = json.str();
    return env->NewStringUTF(payload.c_str());
}
