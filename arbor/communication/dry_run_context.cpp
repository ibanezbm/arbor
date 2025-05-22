#include <string>
#include <vector>
#include <thread>

#include <arbor/spike.hpp>

#include "distributed_context.hpp"
#include "label_resolution.hpp"
#include "util/rangeutil.hpp"

namespace arb {

struct dry_run_context_impl {
    using count_type = typename gathered_vector<spike>::count_type;

    explicit dry_run_context_impl(unsigned num_ranks, unsigned num_cells_per_tile):
        num_ranks_(num_ranks), num_cells_per_tile_(num_cells_per_tile) {};
    std::vector<spike>
    remote_gather_spikes(const std::vector<spike>& local_spikes) const {
        return {};
    }
    gathered_vector<spike>
    all_to_all_spikes(const gathered_vector<spike>& local_spikes) const {
        auto local_size = local_spikes.count(0);
        const auto& local = local_spikes.values();
        std::vector<spike> gathered_spikes;
        gathered_spikes.reserve(static_cast<size_t>(local_size) * static_cast<size_t>(num_ranks_));
        std::vector<count_type> partition;
        partition.reserve(num_ranks_ + 1);
        partition.push_back(0);
        for (std::size_t ridx = 0; ridx < num_ranks_; ridx++) {
            for (std::size_t lidx = 0; lidx < local_size; ++lidx) {
                auto spike = local[lidx];
                spike.source.gid += num_cells_per_tile_*ridx;
                gathered_spikes.push_back(spike);
            }
            partition.push_back(gathered_spikes.size());
        }
        return gathered_vector<spike>(std::move(gathered_spikes), std::move(partition));
    }
    
    gathered_vector<spike>
    all_to_all_buffer_spikes(const std::vector<spike>& local_spikes, 
                             const std::unordered_map<cell_size_type, std::vector<cell_gid_type>>& src_ranks_, 
                             const context& ctx) const {

        constexpr std::size_t buffer_size_bytes = 128 * 1024 * 1024;
        constexpr std::size_t spike_size = sizeof(spike);
        std::size_t spikes_per_buffer = buffer_size_bytes / spike_size;
        std::size_t spikes_per_rank = buffer_size_bytes / spike_size / num_ranks_;
        std::size_t total_buffers = local_spikes.size() * num_ranks_ / spikes_per_buffer;
        
        
        if ((local_spikes.size() * num_ranks_) % (buffer_size_bytes / spike_size)) {
            total_buffers++;
        }
        
        std::vector<std::vector<std::vector<spike>>> rounds(total_buffers,
                                                    std::vector<std::vector<spike>>(num_ranks_));
        
        for (std::size_t b = 0; b < total_buffers; ++b) {
            for (std::size_t r = 0; r < num_ranks_; ++r) {
                rounds[b][r].reserve(spikes_per_rank);
            }
        }
        
        std::vector<bool> ready(total_buffers, false);
        
        std::vector<std::vector<spike>> result(num_ranks_);
        for (std::size_t r = 0; r < num_ranks_; ++r) {
            result[r].reserve(local_spikes.size());
        }
        
        std::vector<count_type> partition(num_ranks_ + 1, 0);
        
        auto receiver = std::thread([&] {
            std::size_t start = 0;
            while (start < total_buffers) {
                while (!ready[start]){std::this_thread::yield();}
                for (std::size_t r = 0; r < num_ranks_; ++r) {
                    result[r].insert(result[r].end(), rounds[start][r].begin(), rounds[start][r].end());
                }
                start++;
            }
        });
        
        std::size_t buffer_id = 0;
        std::size_t spikes_this_round = 0;
        for (const auto& sp : local_spikes) {
            bool sent = false;
            auto it = src_ranks_.find(sp.source.gid);
            if (it != src_ranks_.end()) {
                for (cell_size_type valor : it->second) {
                    auto spike = sp;
                    spike.source.gid += num_cells_per_tile_*valor;
                    rounds[buffer_id][valor].push_back(spike);
                    sent = true;
                }
            }
            if (sent) {
                spikes_this_round++;
                if (spikes_this_round == spikes_per_rank) {
                    ready[buffer_id] = true;
                    buffer_id++;
                    spikes_this_round = 0;
                }
            }
        }
        
        receiver.join();
        
        for (std::size_t r = 1; r <= num_ranks_; ++r) {
            partition[r] = result[r-1].size() + partition[r-1];
        }
        std::vector<spike> flat_result;
        flat_result.reserve(partition.back());
        for (auto& r : result) {
            flat_result.insert(flat_result.end(),
                std::make_move_iterator(r.begin()),
                std::make_move_iterator(r.end()));
        }
        return gathered_vector<spike>(std::move(flat_result), std::move(partition));
    }
    
    gathered_vector<spike>
    gather_spikes(const std::vector<spike>& local_spikes) const {

        count_type local_size = local_spikes.size();

        std::vector<spike> gathered_spikes;
        gathered_spikes.reserve(local_size*num_ranks_);

        for (count_type i = 0; i < num_ranks_; i++) {
            util::append(gathered_spikes, local_spikes);
        }

        for (count_type i = 0; i < num_ranks_; i++) {
            for (count_type j = i*local_size; j < (i+1)*local_size; j++){
                gathered_spikes[j].source.gid += num_cells_per_tile_*i;
            }
        }

        std::vector<count_type> partition;
        for (count_type i = 0; i <= num_ranks_; i++) {
            partition.push_back(static_cast<count_type>(i*local_size));
        }

        return gathered_vector<spike>(std::move(gathered_spikes), std::move(partition));
    }
    void remote_ctrl_send_continue(const epoch&) const {}
    void remote_ctrl_send_done() const {}
    gathered_vector<cell_gid_type>
    gather_gids(const std::vector<cell_gid_type>& local_gids) const {
        count_type local_size = local_gids.size();

        std::vector<cell_gid_type> gathered_gids;
        gathered_gids.reserve(local_size*num_ranks_);

        for (count_type i = 0; i < num_ranks_; i++) {
            util::append(gathered_gids, local_gids);
        }

        for (count_type i = 0; i < num_ranks_; i++) {
            for (count_type j = i*local_size; j < (i+1)*local_size; j++){
                gathered_gids[j] += num_cells_per_tile_*i;
            }
        }

        std::vector<count_type> partition;
        for (count_type i = 0; i <= num_ranks_; i++) {
            partition.push_back(i*local_size);
        }

        return gathered_vector<cell_gid_type>(std::move(gathered_gids), std::move(partition));
    }

    gathered_vector<cell_gid_type>
    all_to_all_gids_domains(const std::vector<std::vector<cell_gid_type>>& gids_domains) const {
        using count_type = gathered_vector<cell_gid_type>::count_type;
        std::size_t local_size = gids_domains[0].size();

        std::vector<cell_gid_type> gathered_gids;
        gathered_gids.reserve(static_cast<size_t>(local_size) * static_cast<size_t>(num_ranks_));
        for (count_type i = 0; i < num_ranks_; i++) {
            util::append(gathered_gids, gids_domains[0]);
        }

        std::vector<count_type> partition;
        for (count_type i = 0; i <= num_ranks_; i++) {
            partition.push_back(static_cast<count_type>(i*local_size));
        }

        return gathered_vector<cell_gid_type>(std::move(gathered_gids), std::move(partition));
    }

    cell_label_range gather_cell_label_range(const cell_label_range& local_ranges) const {
        cell_label_range global_ranges;
        for (unsigned i = 0; i < num_ranks_; i++) {
            global_ranges.append(local_ranges);
        }
        return global_ranges;
    }

    cell_labels_and_gids gather_cell_labels_and_gids(const cell_labels_and_gids& local_labels_and_gids) const {
        auto global_ranges = gather_cell_label_range(local_labels_and_gids.label_range);
        auto gids = gather_gids(local_labels_and_gids.gids);
        return cell_labels_and_gids(global_ranges, gids.values());
    }

    template <typename T>
    std::vector<T> gather(T value, int) const {
        return std::vector<T>(num_ranks_, value);
    }

    std::vector<std::size_t> gather_all(std::size_t value) const {
        return std::vector<std::size_t>(num_ranks_, value);
    }

    distributed_request send_recv_nonblocking(std::size_t dest_count,
        void* dest_data,
        int dest,
        std::size_t source_count,
        const void* source_data,
        int source,
        int tag) const {
        throw arbor_internal_error("send_recv_nonblocking: not implemented for dry run conext.");

        return distributed_request{
            std::make_unique<distributed_request::distributed_request_interface>()};
    }

    int id() const { return 0; }

    int size() const { return num_ranks_; }

    template <typename T>
    T min(T value) const { return value; }

    template <typename T>
    T max(T value) const { return value; }

    template <typename T>
    T sum(T value) const { return value * num_ranks_; }

    void barrier() const {}

    std::string name() const { return "dryrun"; }

    unsigned num_ranks_;
    unsigned num_cells_per_tile_;
};

ARB_ARBOR_API std::shared_ptr<distributed_context> make_dry_run_context(unsigned num_ranks, unsigned num_cells_per_tile) {
    return std::make_shared<distributed_context>(dry_run_context_impl(num_ranks, num_cells_per_tile));
}

} // namespace arb
