#include "../../app/src/main/cpp/tensor_arena_core.h"
#include <cassert>
#include <cstdint>
#include <limits>
using argus::vulkan::TensorArenaCore;
int main() {
    {
        TensorArenaCore arena;
        assert(arena.prepareCapacityExtension() && arena.extendCapacity(0U,256U));
        std::uint64_t first=0U;
        assert(arena.tryAllocate(13U,128U,16U,first));
        const auto* firstItem=arena.find(first);
        assert(firstItem && firstItem->offset==0U && firstItem->logicalSize==13U && firstItem->allocatedSize==16U);
        assert(arena.allocatedBytes()==16U && arena.activeAllocationCount()==1U);
        std::uint64_t second=0U;
        assert(arena.tryAllocate(17U,128U,32U,second));
        const auto* secondItem=arena.find(second);
        assert(secondItem && secondItem->offset==32U && secondItem->allocatedSize==20U);
        assert(arena.release(first) && arena.find(first)==nullptr && !arena.release(first));
        std::uint64_t replacement=0U;
        assert(arena.tryAllocate(12U,128U,16U,replacement));
        const auto* replacementItem=arena.find(replacement);
        assert(replacementItem && replacementItem->offset==0U && replacement!=first);
        assert(arena.release(second) && arena.release(replacement));
        assert(arena.allocatedBytes()==0U && arena.activeAllocationCount()==0U);
        std::uint64_t whole=0U;
        assert(arena.tryAllocate(256U,256U,4U,whole) && arena.find(whole));
    }
    {
        TensorArenaCore previousRuntime;
        assert(previousRuntime.prepareCapacityExtension() && previousRuntime.extendCapacity(0U,64U));
        std::uint64_t stale=0U;
        assert(previousRuntime.tryAllocate(24U,64U,4U,stale));
        previousRuntime.reset();
        TensorArenaCore recreatedRuntime;
        assert(recreatedRuntime.prepareCapacityExtension() && recreatedRuntime.extendCapacity(0U,64U));
        std::uint64_t fresh=0U;
        assert(recreatedRuntime.tryAllocate(24U,64U,4U,fresh));
        assert(fresh!=stale && recreatedRuntime.find(stale)==nullptr && recreatedRuntime.find(fresh));
    }
    {
        TensorArenaCore arena;
        assert(arena.prepareCapacityExtension() && arena.extendCapacity(0U,32U));
        std::uint64_t existing=0U;
        assert(arena.tryAllocate(16U,32U,4U,existing));
        const auto* before=arena.find(existing);
        assert(before);
        const std::uint64_t originalOffset=before->offset;
        assert(arena.prepareCapacityExtension() && arena.extendCapacity(32U,96U));
        const auto* after=arena.find(existing);
        assert(after && after->offset==originalOffset && after->logicalSize==16U);
        std::uint64_t extension=0U;
        assert(arena.tryAllocate(60U,64U,4U,extension));
        const auto* extensionItem=arena.find(extension);
        assert(extensionItem && extensionItem->offset>=originalOffset+16U);
    }
    {
        TensorArenaCore arena;
        assert(arena.prepareCapacityExtension() && arena.extendCapacity(0U,8U));
        std::uint64_t handle=0U;
        assert(!arena.tryAllocate(9U,8U,4U,handle) && handle==0U);
        assert(!arena.tryAllocate(0U,8U,4U,handle));
        assert(!arena.tryAllocate(4U,8U,0U,handle));
        std::uint64_t aligned=0U;
        assert(TensorArenaCore::alignUp(17U,16U,aligned) && aligned==32U);
        assert(!TensorArenaCore::alignUp(std::numeric_limits<std::uint64_t>::max(),16U,aligned));
        assert(!TensorArenaCore::alignUp(1U,0U,aligned));
        assert(!arena.extendCapacity(7U,16U) && !arena.extendCapacity(8U,8U));
    }
    return 0;
}
