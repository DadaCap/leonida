#pragma once

#include "xg_xs_layout.h"
#include "../src/vulkan_backend.h"

#include <algorithm>
#include <deque>
#include <utility>
#include <vector>

namespace rt::d3d12 {

// Upload buffers and command buffers of a device, used under title_upload_mutex.
// A whole image (including arrays and compressed mip tails) is detiled into the mapped buffer of a free slot.
// The transfer never re-enters the D3D12 queue's pre-submit callbacks.
//
// A write no longer waits for its own transfer: the caller (a title thread creating a slot-70 view) waited for the
// GPU at every texture fill, behind all earlier work on the queue, with tracker_mutex_ and submit_mutex_ held, so
// every other submission waited too (speed-26: a title thread 68-88 % inside slot 70). A slot's buffer is reused,
// resized or freed only after its last transfer completed (the slot's timeline value). The image needs no CPU wait:
// later submissions on the queue are ordered after the transfer, other queues wait for it on the GPU, and a retired
// image is destroyed through the deferred queue, after later submissions retire. With no slot free a write is refused
// (room(), `busy`) and its caller leaves the texture for a later pass: a slot's transfer sits behind all the GPU work
// before it, seconds of it at times, and a wait for one holds up the thread that submits.
class TitleImageUpload {
public:
    VkDeviceSize capacity() const {
        VkDeviceSize total = 0;
        for (const auto& slot : slots_) total += slot.capacity;
        return total;
    }

    template<class Device>
    void release(Device& d) {
        auto& vk = d.context->vk;
        wait_all(d);
        for (auto& slot : slots_) release_buffer(d, slot);
        if (pool_) vk.vkDestroyCommandPool(d.info.device, pool_, nullptr);
        if (ready_) vk.vkDestroySemaphore(d.info.device, ready_, nullptr);
        pool_ = VK_NULL_HANDLE;
        for (auto& slot : slots_) slot.command = VK_NULL_HANDLE;
        ready_ = VK_NULL_HANDLE;
    }

    // Uploads the mips from `from` on; `tiled` then needs only mip_chain_bytes(layout, from) bytes.
    template<class Device>
    bool write(Device& d, leonida::ResourceId id, const leonida::ResourceBinding& image,
               const xg_xs::Texture& layout, const std::uint8_t* tiled, std::vector<std::uint64_t>& columns,
               bool inspect, bool& nonzero, bool& busy, unsigned from = 0) {
        busy = false;
        const auto block = leonida::copy_block_format(image, VK_IMAGE_ASPECT_COLOR_BIT);
        const auto mips = layout.layout.mips;
        const auto layers = layout.array;
        if (!block.bytes || block.bytes != layout.format.bytes || from >= mips || mips > image.mip_levels ||
            !layers || layers > image.array_layers) return false;

        regions_.clear();
        VkDeviceSize bytes = 0;
        for (unsigned layer = 0; layer < layers; ++layer) {
            for (unsigned level = from; level < mips; ++level) {
                const auto& mip = layout.layout.plane[0].mips[level];
                VkBufferImageCopy copy{};
                copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1};
                copy.imageExtent = {std::max(1u, image.width >> level), std::max(1u, image.height >> level),
                                    image.volume ? std::max(1u, image.depth >> level) : 1u};
                if (mip.width_elements != (copy.imageExtent.width + block.block - 1) / block.block ||
                    mip.height_elements != (copy.imageExtent.height + block.block - 1) / block.block ||
                    (image.volume && mip.depth != copy.imageExtent.depth)) return false;
                // Vulkan requires offsets aligned to four bytes and to the texel block size.
                const auto align = 4u * block.bytes;
                copy.bufferOffset = bytes + (align - bytes % align) % align;
                const VkDeviceSize row = VkDeviceSize(mip.width_elements) * block.bytes;
                const VkDeviceSize slice = row * mip.height_elements;
                if (slice > UINT32_MAX || (UINT64_MAX - copy.bufferOffset) / copy.imageExtent.depth < slice) return false;
                bytes = copy.bufferOffset + slice * copy.imageExtent.depth;
                regions_.push_back(copy);
            }
        }
        if (!make_ready(d)) return false;
        Slot* const free = pick(d, bytes);
        if (!free) {
            busy = true;
            return false;
        }
        Slot& slot = *free;
        if (!prepare(d, slot, bytes)) return false;
        for (const auto& copy : regions_) {
            const auto level = copy.imageSubresource.mipLevel;
            const auto& mip = layout.layout.plane[0].mips[level];
            const auto row = mip.width_elements * block.bytes, slice = row * mip.height_elements;
            auto* dst = slot.mapped + copy.bufferOffset;
            xg_xs::copy_elements<false>(layout, level, copy.imageSubresource.baseArrayLayer, tiled, dst, row, slice, columns);
            if (inspect && !nonzero)
                nonzero = std::any_of(dst, dst + VkDeviceSize(slice) * copy.imageExtent.depth,
                                      [](std::uint8_t c) { return c != 0; });
        }

        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, from, mips - from, 0, layers};
        const auto result = d.context->write_staged_title_image(id, range, d.info.queue, [&](
            const leonida::ReplayFunctions& replay, const leonida::LayoutPieces& pieces,
            const std::vector<std::pair<VkSemaphore, std::uint64_t>>& earlier) {
            auto& vk = d.context->vk;
            VkResult r = vk.vkResetCommandBuffer(slot.command, 0);
            if (r != VK_SUCCESS) return r;
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if ((r = vk.vkBeginCommandBuffer(slot.command, &begin)) != VK_SUCCESS) return r;
            // All texels of every mip/layer named in range are overwritten. The CPU no longer waits for a transfer,
            // so the barrier orders this one after all earlier work on the queue: earlier reads of the image and an
            // earlier transfer into it, still running.
            leonida::detail::image_barrier(replay, slot.command, image, range, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            vk.vkCmdCopyBufferToImage(slot.command, slot.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      static_cast<std::uint32_t>(regions_.size()), regions_.data());
            for (const auto& piece : pieces)
                leonida::detail::image_barrier(replay, slot.command, image, piece.range, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    piece.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : piece.layout,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            if ((r = vk.vkEndCommandBuffer(slot.command)) != VK_SUCCESS) return r;
            return submit(d, slot, earlier);
        });
        return result == VK_SUCCESS;
    }

    // Whether a write can start now, before its caller reads the texture's bytes for it.
    template<class Device>
    bool room(Device& d) {
        poll(d);
        return std::any_of(slots_.begin(), slots_.end(), [](const Slot& s) { return !s.value; }) ||
               (slots_.size() < kSlots && capacity() < kBytes);
    }

private:
    static constexpr std::size_t kSlots = 32;
    static constexpr VkDeviceSize kBytes = VkDeviceSize(256) << 20;  // of buffers, but one texture may be larger
    struct Slot {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        std::uint8_t* mapped = nullptr;
        VkDeviceSize capacity = 0;
        VkCommandBuffer command{};
        std::uint64_t value = 0;  // the ready_ value its last transfer signals; 0: none submitted
    };

    // The slots whose transfer completed are free.
    template<class Device>
    void poll(Device& d) {
        std::uint64_t reached = 0;
        if (!ready_ || d.context->vk.vkGetSemaphoreCounterValue(d.info.device, ready_, &reached) != VK_SUCCESS) return;
        for (Slot& slot : slots_)
            if (slot.value <= reached) slot.value = 0;
    }

    // A free slot for `size` bytes: the smallest buffer that holds them, else a new slot, else a free one to be given a
    // larger buffer (the other free ones give theirs up past kBytes). Null when none is free and no more may be made.
    template<class Device>
    Slot* pick(Device& d, VkDeviceSize size) {
        poll(d);
        Slot *fit = nullptr, *largest = nullptr;
        for (Slot& slot : slots_) {
            if (slot.value) continue;
            if (slot.capacity >= size && (!fit || slot.capacity < fit->capacity)) fit = &slot;
            if (!largest || slot.capacity > largest->capacity) largest = &slot;
        }
        if (fit) return fit;
        if (slots_.size() < kSlots && (slots_.empty() || capacity() + size <= kBytes)) return &slots_.emplace_back();
        if (largest && capacity() - largest->capacity + size > kBytes)
            for (Slot& slot : slots_)
                if (!slot.value && &slot != largest) release_buffer(d, slot);
        return largest;
    }

    // Waits until the slot's last transfer has completed: only to release it.
    template<class Device>
    bool wait_slot(Device& d, Slot& slot) {
        if (!slot.value || !ready_) return true;
        auto& vk = d.context->vk;
        std::uint64_t reached = 0;
        if (vk.vkGetSemaphoreCounterValue && vk.vkGetSemaphoreCounterValue(d.info.device, ready_, &reached) == VK_SUCCESS &&
            reached >= slot.value) {
            slot.value = 0;
            return true;
        }
        VkSemaphoreWaitInfo done{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        done.semaphoreCount = 1;
        done.pSemaphores = &ready_;
        done.pValues = &slot.value;
        if (vk.vkWaitSemaphores(d.info.device, &done, UINT64_MAX) != VK_SUCCESS) return false;
        slot.value = 0;
        return true;
    }

    template<class Device>
    void wait_all(Device& d) {
        for (auto& slot : slots_) wait_slot(d, slot);
    }

    template<class Device>
    void release_buffer(Device& d, Slot& slot) {
        auto& vk = d.context->vk;
        if (slot.buffer) vk.vkDestroyBuffer(d.info.device, slot.buffer, nullptr);
        if (slot.memory) vk.vkFreeMemory(d.info.device, slot.memory, nullptr);
        slot.buffer = VK_NULL_HANDLE;
        slot.memory = VK_NULL_HANDLE;
        slot.mapped = nullptr;
        slot.capacity = 0;
    }

    template<class Device>
    bool make_ready(Device& d) {
        auto& vk = d.context->vk;
        const auto dev = d.info.device;
        if (!pool_) {
            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.queueFamilyIndex = d.info.queue_family;
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            if (vk.vkCreateCommandPool(dev, &pool, nullptr, &pool_) != VK_SUCCESS) return false;
        }
        if (!ready_) {
            VkSemaphoreTypeCreateInfo timeline{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &timeline};
            if (vk.vkCreateSemaphore(dev, &info, nullptr, &ready_) != VK_SUCCESS) return false;
            for (const auto* qs : {&d.info.command_queues, &d.info.compute_queues, &d.info.copy_queues})
                for (auto q : *qs)
                    if (q && q != d.info.queue && std::find(consumers_.begin(), consumers_.end(), q) == consumers_.end())
                        consumers_.push_back(q);
        }
        return true;
    }

    // `slot` is free: its last transfer is done.
    template<class Device>
    bool prepare(Device& d, Slot& slot, VkDeviceSize size) {
        auto& vk = d.context->vk;
        const auto dev = d.info.device;
        if (!slot.command) {
            VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            alloc.commandPool = pool_;
            alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc.commandBufferCount = 1;
            if (vk.vkAllocateCommandBuffers(dev, &alloc, &slot.command) != VK_SUCCESS) return false;
        }
        if (size <= slot.capacity) return true;
        release_buffer(d, slot);
        VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer.size = size;
        buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vk.vkCreateBuffer(dev, &buffer, nullptr, &slot.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        vk.vkGetBufferMemoryRequirements(dev, slot.buffer, &req);
        slot.memory = d.allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
        void* map = nullptr;
        if (!slot.memory || vk.vkBindBufferMemory(dev, slot.buffer, slot.memory, 0) != VK_SUCCESS ||
            vk.vkMapMemory(dev, slot.memory, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS) {
            release_buffer(d, slot);
            return false;
        }
        slot.mapped = static_cast<std::uint8_t*>(map);
        slot.capacity = size;
        return true;
    }

    // Under tracker_mutex_ and submit_mutex_ (write_staged_title_image): submits the transfer and the other queues'
    // GPU-side waits for it, and returns without waiting for it on the CPU.
    template<class Device>
    VkResult submit(Device& d, Slot& slot, const std::vector<std::pair<VkSemaphore, std::uint64_t>>& earlier) {
        auto& vk = d.context->vk;
        ++value_;
        std::vector<VkSemaphore> after;
        std::vector<std::uint64_t> after_values;
        for (const auto& [progress, value] : earlier) {
            after.push_back(progress);
            after_values.push_back(value);
        }
        const std::vector<VkPipelineStageFlags> after_stages(after.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        timeline.waitSemaphoreValueCount = std::uint32_t(after.size());
        timeline.pWaitSemaphoreValues = after_values.data();
        timeline.signalSemaphoreValueCount = 1;
        timeline.pSignalSemaphoreValues = &value_;
        VkSubmitInfo transfer{VK_STRUCTURE_TYPE_SUBMIT_INFO, &timeline};
        transfer.waitSemaphoreCount = std::uint32_t(after.size());
        transfer.pWaitSemaphores = after.data();
        transfer.pWaitDstStageMask = after_stages.data();
        transfer.commandBufferCount = 1;
        transfer.pCommandBuffers = &slot.command;
        transfer.signalSemaphoreCount = 1;
        transfer.pSignalSemaphores = &ready_;
        // Through the compile gate, as every other submission: it waits here, not in the driver, while a pipeline
        // create holds it, and the caller observes a device loss from it.
        auto r = d.context->queue_submit(d.info.queue, 1, &transfer, VK_NULL_HANDLE);
        if (r != VK_SUCCESS) {
            --value_;  // nothing signalled it
            return r;
        }
        // No CPU wait: the slot's buffer is reused only after this value (wait_slot), the image's later reads are
        // ordered behind the transfer on the GPU, and a retired image is destroyed through the deferred queue.
        slot.value = value_;
        // GPU writes become visible to later submissions on every other queue. The primary queue has the image barrier.
        VkTimelineSemaphoreSubmitInfo acquire{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        acquire.waitSemaphoreValueCount = 1;
        acquire.pWaitSemaphoreValues = &value_;
        VkPipelineStageFlags stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo wait{VK_STRUCTURE_TYPE_SUBMIT_INFO, &acquire};
        wait.waitSemaphoreCount = 1;
        wait.pWaitSemaphores = &ready_;
        wait.pWaitDstStageMask = &stages;
        for (auto q : consumers_)
            if ((r = d.context->queue_submit(q, 1, &wait, VK_NULL_HANDLE)) != VK_SUCCESS) return r;
        return VK_SUCCESS;
    }

    std::deque<Slot> slots_;  // a slot keeps its address
    VkCommandPool pool_{};
    VkSemaphore ready_{};
    std::uint64_t value_ = 0;
    std::vector<VkQueue> consumers_;
    std::vector<VkBufferImageCopy> regions_;
};

}  // namespace rt::d3d12
