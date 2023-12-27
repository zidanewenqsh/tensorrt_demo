#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i<threads; ++i)
            workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers)
            worker.join();
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared< std::packaged_task<return_type()> >(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );
            
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            // don't allow enqueueing after stopping the pool
            if(stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

private:
    std::vector<std::thread> workers;
    std::queue< std::function<void()> > tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// 使用 ThreadPool
// ThreadPool pool(4);
// pool.enqueue([]{ /* Your Task */ });
#if 0
int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}
#include <iostream>
#include <vector>

// 假设 ThreadPool 类和 fibonacci 函数已经定义

int main() {
    ThreadPool pool(4); // 创建包含4个工作线程的线程池

    std::vector< std::future<int> > results;

    // 提交多个任务到线程池
    for(int i = 0; i < 8; ++i) {
        results.emplace_back(
            pool.enqueue(fibonacci, i)
        );
    }

    // 输出任务结果
    for(auto && result: results) {
        std::cout << "Fibonacci: " << result.get() << std::endl;
    }

    return 0;
}
#else
#include <iostream>
#include <vector>
class MyClass {
public:
    int memberFunction(int x) {
        try {
            // 执行一些操作
            // 例如：打印 x 的平方
            std::cout << "The square of " << x << " is " << x * x << std::endl;
            return 0; // 成功
        } catch (...) {
            return -1; // 失败
        }
    }
};

int main() {
    ThreadPool pool(4); // 创建一个包含4个工作线程的线程池
    MyClass myObject; // 创建 MyClass 的一个实例

    std::vector<std::future<int>> results;

    // 提交成员函数到线程池并收集结果
    for(int i = 0; i < 64; ++i) {
        auto result = pool.enqueue(&MyClass::memberFunction, &myObject, i);
        results.push_back(std::move(result));
    }

    // 检查所有任务的结果
    for(auto &result : results) {
        int status = result.get();
        if (status == -1) {
            std::cout << "A task failed." << std::endl;
            // 处理失败情况
        }
    }

    std::cout << "All tasks completed." << std::endl;

    return 0;
}


#endif