// Written in 2020 by Conor Williams (cw648@cam.ac.uk)

#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <ratio>
#include <thread>
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
    template <typename Callback>
    explicit Thiefpool(Callback &&callback, std::size_t num_threads = std::thread::hardware_concurrency())
        : _deques(num_threads), _num_threads(num_threads) {
        if (_num_threads == 0) _num_threads = 1;
        for (std::size_t i = 0; i < _num_threads; ++i) {
            _threads.emplace_back(
                [&, id = i, callback = std::forward<Callback>(callback)](std::stop_token tok) mutable {
                    std::forward<Callback>(callback)([&] { this->start_thread_loop(id, tok); });
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

    auto thread_count() const -> std::size_t { return _num_threads; }

    ~Thiefpool() {
        for (auto &t : _threads) {
            t.request_stop();
        }
        for (auto &d : _deques) {
            d.sem.release();
        }
    }

  private:
    void start_thread_loop(std::size_t id, std::stop_token const &tok) {
        jump(id);  // Get a different random stream
        do {
            // Wait to be signalled
            _deques[id].sem.acquire_many();

            std::size_t spin = 0;

            do {
                // Prioritise our work otherwise steal
                std::size_t t
                    = spin++ < 100 || !_deques[id].tasks.empty() ? id : xoroshiro128() % _deques.size();

                if (std::optional one_shot = _deques[t].tasks.steal()) {
                    _in_flight.fetch_sub(1, std::memory_order_release);
                    std::invoke(std::move(*one_shot), t);
                }

                // Loop until all the work is done.
            } while (_in_flight.load(std::memory_order_acquire) > 0);

        } while (!tok.stop_requested());
    }

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
    std::size_t _num_threads;
};

}  // namespace riften
