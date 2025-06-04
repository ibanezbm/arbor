#pragma once

#include <algorithm>
#include <iostream>
#include <type_traits>
#include <vector>
#include <cstddef>
#include <limits>

#include <mpi.h>

#include <arbor/export.hpp>
#include <arbor/assert.hpp>
#include <arbor/communication/mpi_error.hpp>

#include "communication/gathered_vector.hpp"
#include "profile/profiler_macro.hpp"
#include "util/rangeutil.hpp"
#include "util/partition.hpp"

namespace arb {
namespace mpi {

// prototypes
ARB_ARBOR_API int rank(MPI_Comm);
ARB_ARBOR_API int size(MPI_Comm);
ARB_ARBOR_API void barrier(MPI_Comm);

#define MPI_OR_THROW(fn, ...)\
while (int r_ = fn(__VA_ARGS__)) throw mpi_error(r_, #fn)

// Type traits for automatically setting MPI_Datatype information for C++ types.
template <typename T>
struct mpi_traits {
    constexpr static size_t count() {
        return sizeof(T);
    }
    constexpr static MPI_Datatype mpi_type() {
        return MPI_CHAR;
    }
    constexpr static bool is_mpi_native_type() {
        return false;
    }
};

#define MAKE_TRAITS(T,M)                                        \
template <>                                                     \
struct mpi_traits<T> {                                          \
    constexpr static size_t count()            { return 1; }    \
    static MPI_Datatype mpi_type()   { return M; }              \
    constexpr static bool is_mpi_native_type() { return true; } \
};

MAKE_TRAITS(float,              MPI_FLOAT)
MAKE_TRAITS(double,             MPI_DOUBLE)
MAKE_TRAITS(char,               MPI_CHAR)
MAKE_TRAITS(int,                MPI_INT)
MAKE_TRAITS(unsigned,           MPI_UNSIGNED)
MAKE_TRAITS(long,               MPI_LONG)
MAKE_TRAITS(unsigned long,      MPI_UNSIGNED_LONG)
MAKE_TRAITS(long long,          MPI_LONG_LONG)
MAKE_TRAITS(unsigned long long, MPI_UNSIGNED_LONG_LONG)

static_assert(std::is_same<std::size_t, unsigned>::value ||
              std::is_same<std::size_t, unsigned long>::value ||
              std::is_same<std::size_t, unsigned long long>::value,
              "size_t is not the same as any MPI unsigned type");

// Gather individual values of type T from each rank into a std::vector on
// the root rank.
// T must be trivially copyable.
template<typename T>
std::vector<T> gather(T value, int root, MPI_Comm comm) {
    using traits = mpi_traits<T>;
    auto buffer_size = (rank(comm)==root) ? size(comm) : 0;
    std::vector<T> buffer(buffer_size);

    MPI_OR_THROW(MPI_Gather,
                &value,        traits::count(), traits::mpi_type(), // send buffer
                buffer.data(), traits::count(), traits::mpi_type(), // receive buffer
                root, comm);

    return buffer;
}

// Gather individual values of type T from each rank into a std::vector on
// the every rank.
// T must be trivially copyable
template <typename T>
std::vector<T> gather_all(T value, MPI_Comm comm) {
    using traits = mpi_traits<T>;
    std::vector<T> buffer(size(comm));

    MPI_OR_THROW(MPI_Allgather,
            &value,        traits::count(), traits::mpi_type(), // send buffer
            buffer.data(), traits::count(), traits::mpi_type(), // receive buffer
            comm);

    return buffer;
}

// Specialize gather for std::string.
inline std::vector<std::string> gather(std::string str, int root, MPI_Comm comm) {
    using traits = mpi_traits<char>;

    std::vector<int> counts, displs;
    counts = gather_all(int(str.size()), comm);
    util::make_partition(displs, counts);

    std::vector<char> buffer(displs.back());

    // const_cast required for MPI implementations that don't use const* in
    // their interfaces.
    std::string::value_type* ptr = const_cast<std::string::value_type*>(str.data());
    MPI_OR_THROW(MPI_Gatherv,
            ptr, counts[rank(comm)], traits::mpi_type(),                       // send
            buffer.data(), counts.data(), displs.data(), traits::mpi_type(),   // receive
            root, comm);

    // Unpack the raw string data into a vector of strings.
    std::vector<std::string> result;
    auto nranks = size(comm);
    result.reserve(nranks);
    for (auto i=0; i<nranks; ++i) {
        result.push_back(std::string(buffer.data()+displs[i], counts[i]));
    }
    return result;
}

template <typename T>
std::vector<T> gather_all(const std::vector<T>& values, MPI_Comm comm) {
    using traits = mpi_traits<T>;
    auto counts = gather_all(int(values.size()), comm);
    for (auto& c : counts) c *= traits::count();
    std::vector<int> displs;
    util::make_partition(displs, counts);
    std::vector<T> buffer(displs.back()/traits::count());
    auto send_count = values.size()*traits::count();
    MPI_OR_THROW(MPI_Allgatherv,
                 // const_cast required for MPI implementations that don't use const* in their interfaces
                 const_cast<T*>(values.data()), send_count, traits::mpi_type(),  // send buffer
                 buffer.data(), counts.data(), displs.data(), traits::mpi_type(), // receive buffer
                 comm);
    return buffer;
}

inline std::vector<std::string> gather_all(const std::vector<std::string>& values, MPI_Comm comm) {
    using traits = mpi_traits<char>;
    std::vector<int> counts_individual, counts_total, displs_individual, displs_total;

    // vector of individual string sizes
    std::vector<int> individual_sizes(values.size());
    std::transform(values.begin(), values.end(), individual_sizes.begin(), [](const std::string& val){return int(val.size());});

    counts_individual = gather_all(individual_sizes, comm);
    counts_total      = gather_all(util::sum(individual_sizes, 0), comm);

    util::make_partition(displs_total, counts_total);
    std::vector<char> buffer(displs_total.back());

    // Concatenate string data
    std::string values_concat;
    for (const auto& v: values) {
        values_concat += v;
    }

    // Cast to ptr
    // const_cast required for MPI implementations that don't use const* in
    // their interfaces.
    std::string::value_type* ptr = const_cast<std::string::value_type*>(values_concat.data());
    MPI_OR_THROW(MPI_Allgatherv,
                 ptr, counts_total[rank(comm)], traits::mpi_type(),  // send buffer
                 buffer.data(), counts_total.data(), displs_total.data(), traits::mpi_type(), // receive buffer
                 comm);

    // Construct the vector of strings
    std::vector<std::string> string_buffer;
    string_buffer.reserve(counts_individual.size());

    auto displs_individual_part = util::make_partition(displs_individual, counts_individual);
    for (const auto& str_range: displs_individual_part) {
        string_buffer.emplace_back(buffer.begin()+str_range.first, buffer.begin()+str_range.second);
    }

    return string_buffer;
}

template <typename T>
std::vector<std::vector<T>> gather_all(const std::vector<std::vector<T>>& values, MPI_Comm comm) {
    std::vector<unsigned long> counts_internal, displs_internal;

    // Vector of individual vector sizes
    std::vector<unsigned long> internal_sizes(values.size());
    std::transform(values.begin(), values.end(), internal_sizes.begin(), [](const auto& val){return int(val.size());});

    counts_internal = gather_all(internal_sizes, comm);
    auto displs_internal_part = util::make_partition(displs_internal, counts_internal);

    // Concatenate all internal vector data
    std::vector<T> values_concat;
    for (const auto& v: values) {
        values_concat.insert(values_concat.end(), v.begin(), v.end());
    }

    // Gather all concatenated vector data
    auto global_vec_concat = gather_all(values_concat, comm);

    // Construct the vector of vectors
    std::vector<std::vector<T>> global_vec;
    global_vec.reserve(displs_internal_part.size());

    for (const auto& internal_vec_range: displs_internal_part) {
        global_vec.emplace_back(global_vec_concat.begin()+internal_vec_range.first,
                                global_vec_concat.begin()+internal_vec_range.second);
    }

    return global_vec;
}

/// Gather all of a distributed vector
/// Retains the meta data (i.e. vector partition)
template <typename T>
gathered_vector<T> gather_all_with_partition(const std::vector<T>& values, MPI_Comm comm) {
    using gathered_type = gathered_vector<T>;
    using count_type = typename gathered_vector<T>::count_type;
    using traits = mpi_traits<T>;

    // We have to use int for the count and displs vectors instead
    // of count_type because these are used as arguments to MPI_Allgatherv
    // which expects int arguments.
    std::vector<int> counts, displs;
    counts = gather_all(int(values.size()), comm);
    for (auto& c : counts) {
        c *= traits::count();
    }
    util::make_partition(displs, counts);

    std::vector<T> buffer(displs.back()/traits::count());

    MPI_OR_THROW(MPI_Allgatherv,
            // const_cast required for MPI implementations that don't use const* in their interfaces
            const_cast<T*>(values.data()), counts[rank(comm)], traits::mpi_type(), // send buffer
            buffer.data(), counts.data(), displs.data(), traits::mpi_type(), // receive buffer
            comm);

    for (auto& d : displs) {
        d /= traits::count();
    }

    return gathered_type(
        std::move(buffer),
        std::vector<count_type>(displs.begin(), displs.end())
    );
}

template <typename T>
inline gathered_vector<T> all_to_all_impl(const std::vector<T>& send_buffer, const std::vector<int>& send_counts,
                                          const std::vector<int>& send_displs, int num_ranks, MPI_Comm comm){
    using gathered_type = gathered_vector<T>;
    using count_type = typename gathered_type::count_type;
    using traits = mpi_traits<T>;

    std::vector<int> recv_counts(num_ranks, 0);
    std::vector<int> recv_displs(num_ranks, 0);

    MPI_OR_THROW(MPI_Alltoall,
                 send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 comm);

    util::make_partition(recv_displs, recv_counts);

    auto count_per_element = traits::count();
    std::vector<T> recv_buffer(recv_displs.back() / count_per_element);

    MPI_OR_THROW(MPI_Alltoallv,
                 const_cast<T*>(send_buffer.data()), send_counts.data(), send_displs.data(), traits::mpi_type(),
                 recv_buffer.data(), recv_counts.data(), recv_displs.data(), traits::mpi_type(),
                 comm);

    for (auto& d : recv_displs) {
        d /= count_per_element;
    }

    std::vector<count_type> partition;
    partition.reserve(recv_displs.size());
    std::transform(recv_displs.begin(), recv_displs.end(), std::back_inserter(partition),
                   [](int v) { return static_cast<count_type>(v); });

    return gathered_type(std::move(recv_buffer), std::move(partition));
}

inline void wait_all(std::vector<MPI_Request> &&requests) {
    if(!requests.empty()) {
        MPI_OR_THROW(
            MPI_Waitall, static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
}

template <typename T>
inline gathered_vector<T> all_to_all_impl_batched(const std::vector<T>& send_buffer,
                                                  const std::vector<int>& send_counts,
                                                  const std::vector<int>& send_displs,
                                                  int num_ranks,
                                                  MPI_Comm comm,
                                                  std::size_t batch_bytes = 128 * 1024 * 1024) {
    using gathered_type = gathered_vector<T>;
    using count_type = typename gathered_type::count_type;
    using traits = mpi_traits<T>;
    std::vector<count_type> partition(num_ranks + 1, 0);
    std::vector<T> recv_buffer;

    std::vector<int> rounds_per_rank(num_ranks, 0);
    int spikes_per_rank = batch_bytes / traits::count() / num_ranks;
    
    for (int i = 0; i < num_ranks; i++){
        rounds_per_rank[i] = static_cast<std::size_t>(std::ceil(
                                                      static_cast<double>(send_counts[i]) / spikes_per_rank));
    }

    int rounds = *std::max_element(rounds_per_rank.begin(), rounds_per_rank.end());
    int rounds_max;

    PE(communication:exchange:all2all:allreduce);
    MPI_Allreduce(&rounds, &rounds_max, 1, MPI_INT, MPI_MAX, comm);
    PL();
    std::vector<int> spikes_sent_per_rank(num_ranks, 0);
    int cur_round = 0;
    while (cur_round < rounds_max) {
        std::vector<T> send_buffer_round;
        std::vector<int> send_counts_round(num_ranks, 0);
        std::vector<int> send_displs_round(num_ranks, 0);
        
        PE(communication:exchange:all2all:prev);
        send_buffer_round.reserve(batch_bytes / traits::count());
	for (int rank = 0; rank < num_ranks; rank++) {
	    const int spikes_sent = spikes_sent_per_rank[rank];
            const int send_count = send_counts[rank];
            if(spikes_sent < send_count) {
                
                std::size_t sent_spikes_init = send_displs[rank] + spikes_sent;
                std::size_t sent_spikes_end = sent_spikes_init + std::min(spikes_per_rank, 
                                                                          send_count - spikes_sent);
		std::size_t batch_size = sent_spikes_end - sent_spikes_init;
                //printf("AAAAAA %ld %ld %d %d\n",sent_spikes_init,sent_spikes_end, spikes_per_rank, send_counts[rank] - spikes_sent_per_rank[rank]);
		send_buffer_round.insert(send_buffer_round.end(),
                                 send_buffer.begin() + sent_spikes_init,
                                 send_buffer.begin() + sent_spikes_end);
                
		spikes_sent_per_rank[rank] += batch_size;
                send_counts_round[rank] = batch_size * traits::count();
            }
            
            if(rank != 0) {
                send_displs_round[rank] = send_counts_round[rank - 1] + send_displs_round[rank - 1];
            }
        }
        PL();
        std::vector<int> recv_counts_round(num_ranks, 0);
        std::vector<int> recv_displs_round(num_ranks, 0);

        PE(communication:exchange:all2all:sizes);
        MPI_OR_THROW(MPI_Alltoall,
                     send_counts_round.data(), 1, MPI_INT,
                     recv_counts_round.data(), 1, MPI_INT,
                     comm);
        
        util::make_partition(recv_displs_round, recv_counts_round);

        auto count_per_element = traits::count();
        std::vector<T> recv_buffer_round(recv_displs_round.back() / count_per_element);
        
/*            printf("send_counts: ");
for (std::size_t i = 0; i < send_counts_round.size(); ++i) {
    printf("%d ", send_counts_round[i]);
}
printf("\n");

            printf("spikes_sent: ");
for (std::size_t i = 0; i < spikes_sent_per_rank.size(); ++i) {
    printf("%d ", spikes_sent_per_rank[i]);
}
printf("\n");

printf("send_displs: ");
for (std::size_t i = 0; i < send_displs_round.size(); ++i) {
    printf("%d ", send_displs_round[i]);
}

printf("recv_count: ");
for (std::size_t i = 0; i < recv_counts_round.size(); ++i) {
    printf("%d ", recv_counts_round[i]);
}

printf("recv_displs: ");
for (std::size_t i = 0; i < recv_displs_round.size(); ++i) {
    printf("%d ", recv_displs_round[i]);
}

printf("partition: ");
for (std::size_t i = 0; i < partition.size(); ++i) {
    printf("%d ", partition[i]);
}
printf("\n");*/

        PL();

        PE(communication:exchange:all2all:communication);
        MPI_OR_THROW(MPI_Alltoallv,
                                send_buffer_round.data(), send_counts_round.data(), 
                                send_displs_round.data(), traits::mpi_type(),
                                recv_buffer_round.data(), recv_counts_round.data(), 
                                recv_displs_round.data(), traits::mpi_type(),
                                comm);
        PL();


        PE(communication:exchange:all2all:post);
        int total_spikes = 0;
        for (int rank = 0; rank < num_ranks; rank++) {
            total_spikes += partition[rank + 1];
            recv_buffer.insert(recv_buffer.begin() + total_spikes, 
                               recv_buffer_round.begin() + (recv_displs_round[rank] / count_per_element), 
                               recv_buffer_round.begin()+ (recv_displs_round[rank+1] / count_per_element));
            partition[rank + 1] += recv_counts_round[rank] / count_per_element;
            total_spikes += recv_counts_round[rank] / count_per_element;
        }
        cur_round++;
	PL();
    }

    PE(communication:exchange:all2all:partition);
    for (int rank = 1; rank < num_ranks + 1; rank++) {
        partition[rank] += partition[rank - 1];
    }
    PL();
    //printf("AAAA %ld %d\n", recv_buffer.size(), partition.back());
    return gathered_type(std::move(recv_buffer), std::move(partition));
}

/// AlltoAll of a gathered vector
/// Retains the meta data (i.e. vector partition)
template <typename T>
gathered_vector<T> all_to_all_with_partition(const gathered_vector<T>& values, MPI_Comm comm) {
    //using traits = mpi_traits<T>;

    int num_ranks = values.partition().size() - 1;

    std::vector<int> send_counts(num_ranks);
    std::vector<int> send_displs(num_ranks);

    const auto& send_buffer = values.values();
    const auto& partition = values.partition();

    for (int i = 0; i < num_ranks; ++i) {
        int count = values.count(i);
        send_counts[i] = count;
        send_displs[i] = partition[i];
    }

    return all_to_all_impl_batched(send_buffer,
                           send_counts,
                           send_displs,
                           num_ranks,
                           comm);
}

/// AlltoAll of a distributed vector
/// Retains the meta data (i.e. vector partition)
template <typename T>
gathered_vector<T> all_to_all_with_partition(const std::vector<std::vector<T>>& values, MPI_Comm comm) {
    //using traits = mpi_traits<T>;
    int num_ranks = values.size();

    std::vector<int> send_counts(num_ranks, 0);
    std::vector<int> send_displs(num_ranks, 0);
    
    std::size_t offset = 0;
    for (int i = 0; i < num_ranks; ++i) {
        send_counts[i] = static_cast<int>(values[i].size());
        send_displs[i] = static_cast<int>(offset);
        offset += values[i].size();
    }
    
    std::vector<T> send_buffer(offset);

    std::size_t buf_offset = 0;
    for (int i = 0; i < num_ranks; ++i) {
        const auto& vec = values[i];
        std::copy(vec.begin(), vec.end(), send_buffer.begin() + buf_offset);
        buf_offset += vec.size();
    }

    return all_to_all_impl_batched(send_buffer,
                           send_counts,
                           send_displs,
                           num_ranks,
                           comm);    
}


template <typename T>
T reduce(T value, MPI_Op op, int root, MPI_Comm comm) {
    using traits = mpi_traits<T>;
    static_assert(traits::is_mpi_native_type(),
                  "can only perform reductions on MPI native types");

    T result;

    MPI_OR_THROW(MPI_Reduce,
        &value, &result, 1, traits::mpi_type(), op, root, comm);

    return result;
}

template <typename T>
T reduce(T value, MPI_Op op, MPI_Comm comm) {
    using traits = mpi_traits<T>;
    static_assert(traits::is_mpi_native_type(),
                  "can only perform reductions on MPI native types");

    T result;

    MPI_Allreduce(&value, &result, 1, traits::mpi_type(), op, comm);

    return result;
}

template <typename T>
std::pair<T,T> minmax(T value) {
    return {reduce<T>(value, MPI_MIN), reduce<T>(value, MPI_MAX)};
}

template <typename T>
std::pair<T,T> minmax(T value, int root) {
    return {reduce<T>(value, MPI_MIN, root), reduce<T>(value, MPI_MAX, root)};
}

template <typename T>
T broadcast(T value, int root, MPI_Comm comm) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "broadcast can only be performed on trivally copyable types");

    using traits = mpi_traits<T>;

    MPI_OR_THROW(MPI_Bcast,
        &value, traits::count(), traits::mpi_type(), root, comm);

    return value;
}

template <typename T>
T broadcast(int root, MPI_Comm comm) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "broadcast can only be performed on trivally copyable types");

    using traits = mpi_traits<T>;
    T value;

    MPI_OR_THROW(MPI_Bcast,
        &value, traits::count(), traits::mpi_type(), root, comm);

    return value;
}

inline std::vector<MPI_Request> isend(std::size_t num_bytes,
    const void* data,
    int dest,
    int tag,
    MPI_Comm comm) {
    constexpr std::size_t max_msg_size = static_cast<std::size_t>(std::numeric_limits<int>::max());

    std::vector<MPI_Request> requests;

    for (std::size_t idx = 0; idx < num_bytes; idx += max_msg_size) {
        requests.emplace_back();
        MPI_OR_THROW(MPI_Isend,
            reinterpret_cast<char*>(const_cast<void*>(data)) + idx,
            static_cast<int>(std::min(max_msg_size, num_bytes - idx)),
            MPI_BYTE,
            dest,
            tag,
            comm,
            &(requests.back()));
    }

    return requests;
}

inline std::vector<MPI_Request> irecv(std::size_t num_bytes,
    void* data,
    int source,
    int tag,
    MPI_Comm comm) {
    constexpr std::size_t max_msg_size = static_cast<std::size_t>(std::numeric_limits<int>::max());

    std::vector<MPI_Request> requests;

    for (std::size_t idx = 0; idx < num_bytes; idx += max_msg_size) {
        requests.emplace_back();
        MPI_OR_THROW(MPI_Irecv,
            reinterpret_cast<char*>(data) + idx,
            static_cast<int>(std::min(max_msg_size, num_bytes - idx)),
            MPI_BYTE,
            source,
            tag,
            comm,
            &(requests.back()));
    }

    return requests;
}

} // namespace mpi
} // namespace arb

