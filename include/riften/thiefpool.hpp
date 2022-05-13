// Written in 2020 by Conor Williams (cw648@cam.ac.uk)

#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <future>
#include <ratio>
#include <type_traits>
#include <utility>

#include "function2/function2.hpp"
#include "jthread.hpp"
#include "riften/deque.hpp"
#include "semaphore.hpp"
#include "xoroshiro128starstar.hpp"

namespace riften {

// Lightweight, fast, work-stealing thread-pool for C++20. Built on the lock-free concurrent `riften::Deque`.
// Upon destruction the threadpool blocks until all tasks have been completed and all threads have joined.
class Thiefpool {
  public:
    // Construct a `Thiefpool` with `num_threads` threads.
    explicit Thiefpool(std::size_t num_threads = std::thread::hardware_concurrency()) : _deques(num_threads) {
        for (std::size_t i = 0; i < num_threads; ++i) {
            _threads.emplace_back([&, id = i](std::stop_token tok) {
                jump(id);  // Get a different random stream
                do {
                    // Wait to be signalled
                    _deques[id].sem.acquire_many();

                    std::size_t spin = 0;

                    do {
                        // Prioritise our work otherwise steal
                        std::size_t t = spin++ < 100 || !_deques[id].tasks.empty()
                                            ? id
                                            : xoroshiro128() % _deques.size();

                        if (std::optional one_shot = _deques[t].tasks.steal()) {
                            _in_flight.fetch_sub(1, std::memory_order_release);
                            std::invoke(std::move(*one_shot), t);
                        }

                        // Loop until all the work is done.
                    } while (_in_flight.load(std::memory_order_acquire) > 0);

                } while (!tok.stop_requested());
            });
        }
    }

    // Enqueue callable `f` into the threadpool. This version does *not* return a handle to the called
    // function and thus only accepts functions which return void.
    template <typename F> void enqueue(F &&f) {
        // Cleaner error message than concept
        static_assert(std::is_same_v<void, std::invoke_result_t<std::decay_t<F>, size_t>>,
                      "Function must return void.");

        execute(std::forward<F>(f));
    }

    ~Thiefpool() {
        for (auto &t : _threads) {
            t.request_stop();
        }
        for (auto &d : _deques) {
            d.sem.release();
        }
    }

  private:
    // Fire and forget interface.
    template <typename F> void execute(F &&f) {
        std::size_t i = count++ % _deques.size();

        _in_flight.fetch_add(1, std::memory_order_relaxed);
        _deques[i].tasks.emplace(std::forward<F>(f));
        _deques[i].sem.release();
    }

    struct named_pair {
        Semaphore sem{0};
        Deque<fu2::unique_function<void(size_t) &&>> tasks;
    };

    std::atomic<std::int64_t> _in_flight;
    std::size_t count = 0;
    std::vector<named_pair> _deques;
    std::vector<std::jthread> _threads;
};

}  // namespace riften
