// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or
 * other RISCs Any two RISC processors cannot use the same CMD_BUF non_blocking APIs shouldn't be mixed with slow noc.h
 * APIs explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
    std::uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);

    std::uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_bank_id        = get_arg_val<uint32_t>(2);

    std::uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(3);
    std::uint32_t dram_dst_bank_id        = get_arg_val<uint32_t>(4);

    std::uint32_t dram_buffer_size      = get_arg_val<uint32_t>(5);

#if defined(SIGNAL_COMPLETION_TO_DISPATCHER)
    // We will assert later. This kernel will hang.
    // Need to signal completion to dispatcher before hanging so that
    // Dispatcher Kernel is able to finish.
    // Device Close () requires fast dispatch kernels to finish.
#if defined(COMPILE_FOR_ERISC)
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);
#else
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
#endif
    uint64_t dispatch_addr = NOC_XY_ADDR(
        NOC_X(mailboxes->go_message.master_x),
        NOC_Y(mailboxes->go_message.master_y),
        DISPATCH_MESSAGE_ADDR + NOC_STREAM_REG_SPACE_SIZE * mailboxes->go_message.dispatch_message_offset);
    noc_fast_write_dw_inline<DM_DEDICATED_NOC>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        1 << REMOTE_DEST_BUF_WORDS_FREE_INC,
        dispatch_addr,
        0xF,  // byte-enable
        NOC_UNICAST_WRITE_VC,
        false,  // mcast
        true    // posted
    );
#endif

    // DRAM NOC src address
    std::uint64_t dram_buffer_src_noc_addr = get_noc_addr_from_bank_id<true>(dram_src_bank_id, dram_buffer_src_addr);
    noc_async_read(dram_buffer_src_noc_addr, l1_buffer_addr, dram_buffer_size);
    noc_async_read_barrier();

    // DRAM NOC dst address
    std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr_from_bank_id<true>(dram_dst_bank_id, dram_buffer_dst_addr);
    noc_async_write(l1_buffer_addr, dram_buffer_dst_noc_addr, dram_buffer_size);
    noc_async_write_barrier();
}
