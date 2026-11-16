#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>
namespace argus::vulkan {
class TensorArenaCore final {
public:
    struct Allocation final {
        std::uint64_t offset=0U, allocatedSize=0U, logicalSize=0U;
        std::uint32_t generation=0U;
        bool active=false;
    };
    bool prepareCapacityExtension() noexcept {
        try { free_.reserve(free_.size()+1U); return true; }
        catch (const std::bad_alloc&) { return false; }
    }
    bool extendCapacity(std::uint64_t oldSize,std::uint64_t newSize) noexcept {
        if (capacity_!=oldSize || newSize<=oldSize || free_.capacity()<free_.size()+1U) return false;
        const std::uint64_t extension=newSize-oldSize;
        for (const FreeBlock& block:free_) {
            if (overlaps(oldSize,extension,block.offset,block.size)) return false;
        }
        free_.push_back({oldSize,extension});
        capacity_=newSize;
        merge();
        return true;
    }
    bool tryAllocate(std::uint64_t logicalSize,std::uint64_t maxRange,
                     std::uint64_t alignment,std::uint64_t& handle) noexcept {
        handle=0U;
        if (logicalSize==0U || maxRange==0U || alignment==0U || nextGeneration_==0U) return false;
        std::uint64_t size=0U;
        if (!alignUp(logicalSize,4U,size) || size>maxRange ||
            size>std::numeric_limits<std::uint64_t>::max()-allocated_) return false;
        std::size_t blockIndex=free_.size();
        std::uint64_t offset=0U,prefix=0U,suffixOffset=0U,suffix=0U;
        for (std::size_t i=0U;i<free_.size();++i) {
            const FreeBlock block=free_[i];
            std::uint64_t aligned=0U;
            if (!alignUp(block.offset,std::max<std::uint64_t>(4U,alignment),aligned) ||
                aligned<block.offset) continue;
            const std::uint64_t before=aligned-block.offset;
            if (before>block.size || size>block.size-before) continue;
            blockIndex=i; offset=aligned; prefix=before;
            suffixOffset=aligned+size; suffix=block.size-before-size;
            break;
        }
        if (blockIndex==free_.size()) return false;
        std::size_t slot=allocations_.size();
        for (std::size_t i=0U;i<allocations_.size();++i) {
            if (!allocations_[i].active) { slot=i; break; }
        }
        const bool newSlot=slot==allocations_.size();
        if (newSlot && slot>=static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()-1U)) return false;
        try {
            if (prefix>0U && suffix>0U) free_.reserve(free_.size()+1U);
            if (newSlot) allocations_.reserve(allocations_.size()+1U);
        } catch (const std::bad_alloc&) { return false; }
        const FreeBlock original=free_[blockIndex];
        if (prefix>0U && suffix>0U) {
            free_[blockIndex]={original.offset,prefix};
            free_.insert(free_.begin()+static_cast<std::ptrdiff_t>(blockIndex+1U),{suffixOffset,suffix});
        } else if (prefix>0U) free_[blockIndex]={original.offset,prefix};
        else if (suffix>0U) free_[blockIndex]={suffixOffset,suffix};
        else free_.erase(free_.begin()+static_cast<std::ptrdiff_t>(blockIndex));
        if (newSlot) allocations_.emplace_back();
        Allocation& item=allocations_[slot];
        item.offset=offset; item.allocatedSize=size; item.logicalSize=logicalSize;
        item.generation=nextGeneration_; item.active=true;
        if (nextGeneration_==std::numeric_limits<std::uint32_t>::max()) nextGeneration_=0U;
        else ++nextGeneration_;
        allocated_+=size;
        handle=(static_cast<std::uint64_t>(item.generation)<<32U)|(static_cast<std::uint64_t>(slot)+1U);
        return true;
    }
    Allocation* find(std::uint64_t handle) noexcept {
        if (handle==0U) return nullptr;
        const auto encoded=static_cast<std::uint32_t>(handle&0xFFFFFFFFULL);
        const auto generation=static_cast<std::uint32_t>(handle>>32U);
        if (encoded==0U || generation==0U) return nullptr;
        const std::size_t index=static_cast<std::size_t>(encoded-1U);
        if (index>=allocations_.size()) return nullptr;
        Allocation& item=allocations_[index];
        return item.active && item.generation==generation ? &item : nullptr;
    }
    const Allocation* find(std::uint64_t handle) const noexcept {
        if (handle==0U) return nullptr;
        const auto encoded=static_cast<std::uint32_t>(handle&0xFFFFFFFFULL);
        const auto generation=static_cast<std::uint32_t>(handle>>32U);
        if (encoded==0U || generation==0U) return nullptr;
        const std::size_t index=static_cast<std::size_t>(encoded-1U);
        if (index>=allocations_.size()) return nullptr;
        const Allocation& item=allocations_[index];
        return item.active && item.generation==generation ? &item : nullptr;
    }
    bool release(std::uint64_t handle) noexcept {
        Allocation* item=find(handle);
        if (item==nullptr || item->allocatedSize==0U || item->offset>capacity_ ||
            item->allocatedSize>capacity_-item->offset || item->allocatedSize>allocated_) return false;
        const std::uint64_t offset=item->offset,size=item->allocatedSize;
        for (const FreeBlock& block:free_) if (overlaps(offset,size,block.offset,block.size)) return false;
        try { free_.reserve(free_.size()+1U); }
        catch (const std::bad_alloc&) { return false; }
        free_.push_back({offset,size}); merge(); allocated_-=size;
        item->offset=0U; item->allocatedSize=0U; item->logicalSize=0U;
        item->generation=0U; item->active=false;
        return true;
    }
    void reset() noexcept {
        capacity_=0U; allocated_=0U; allocations_.clear(); free_.clear();
        // Static generation is intentionally retained across Vulkan runtime recreation.
    }
    std::uint64_t capacityBytes() const noexcept { return capacity_; }
    std::uint64_t allocatedBytes() const noexcept { return allocated_; }
    std::uint32_t activeAllocationCount() const noexcept {
        std::uint32_t count=0U;
        for (const Allocation& item:allocations_) if (item.active && count<UINT32_MAX) ++count;
        return count;
    }
    static bool alignUp(std::uint64_t value,std::uint64_t alignment,std::uint64_t& output) noexcept {
        if (alignment==0U) return false;
        const std::uint64_t remainder=value%alignment;
        const std::uint64_t padding=remainder==0U ? 0U : alignment-remainder;
        if (value>std::numeric_limits<std::uint64_t>::max()-padding) return false;
        output=value+padding; return true;
    }
private:
    struct FreeBlock final { std::uint64_t offset=0U,size=0U; };
    static bool overlaps(std::uint64_t a,std::uint64_t as,std::uint64_t b,std::uint64_t bs) noexcept {
        if (as==0U || bs==0U) return false;
        const auto max=std::numeric_limits<std::uint64_t>::max();
        if (as>max-a || bs>max-b) return true;
        return a<b+bs && b<a+as;
    }
    void merge() noexcept {
        std::sort(free_.begin(),free_.end(),[](const FreeBlock& a,const FreeBlock& b){return a.offset<b.offset;});
        std::size_t write=0U;
        for (std::size_t read=0U;read<free_.size();++read) {
            const FreeBlock block=free_[read];
            if (block.size==0U) continue;
            if (write>0U) {
                FreeBlock& previous=free_[write-1U];
                if (previous.offset+previous.size==block.offset) { previous.size+=block.size; continue; }
            }
            free_[write++]=block;
        }
        free_.resize(write);
    }
    std::uint64_t capacity_=0U,allocated_=0U;
    std::vector<Allocation> allocations_;
    std::vector<FreeBlock> free_;
    inline static std::uint32_t nextGeneration_=1U;
};
} // namespace argus::vulkan
