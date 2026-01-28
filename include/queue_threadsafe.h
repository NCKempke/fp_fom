#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template<typename T>
class ThreadSafeQueue {
private:
    mutable std::mutex mutex;
    std::queue<T> queue;
    std::condition_variable cond_var;
    bool stopped = false;

public:
    ThreadSafeQueue() = default;

    // Disable copying
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    // Allow moving
    ThreadSafeQueue(ThreadSafeQueue&&) = default;
    ThreadSafeQueue& operator=(ThreadSafeQueue&&) = default;

    // Push an item to the queue
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopped) return;
            queue.push(std::move(item));
        }
        cond_var.notify_one();
    }

    // Try to pop an item (non-blocking)
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty() || stopped) {
            return std::nullopt;
        }
        T item = std::move(queue.front());
        queue.pop();
        return item;
    }

    // Pop an item (blocking until an item is available)
    std::optional<T> wait_and_pop() {
        std::unique_lock<std::mutex> lock(mutex);
        cond_var.wait(lock, [this]() {
            return !queue.empty() || stopped;
        });

        if (stopped && queue.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue.front());
        queue.pop();
        return item;
    }

    // Pop with timeout
    template<typename Rep, typename Period>
    std::optional<T> wait_and_pop_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!cond_var.wait_for(lock, timeout, [this]() {
            return !queue.empty() || stopped;
        })) {
            return std::nullopt; // Timeout
        }

        if (stopped && queue.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue.front());
        queue.pop();
        return item;
    }

    // Check if queue is empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }

    // Get size
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }

    // Stop the queue (wakes up all waiting threads)
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopped = true;
        }
        cond_var.notify_all();
    }

    // Clear the queue
    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        while (!queue.empty()) {
            queue.pop();
        }
    }
};