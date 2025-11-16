#pragma once

#include <hip/hip_runtime.h>

namespace gpu {

template <typename T>
	requires std::is_trivially_copyable_v<T>
class device_unique_ptr;

template <typename T>
	requires std::is_trivially_copyable_v<T>
class device_span {
public:
	using element_type   = T;
	using value_type     = std::remove_cv_t<T>;
	using size_type      = std::size_t;
	using pointer        = T *;
	using const_pointer  = const T *;
	using iterator       = pointer;
	using const_iterator = const_pointer;

protected:
	pointer data_   = nullptr;
	size_type size_ = 0;

public:
	constexpr device_span() noexcept = default;

	constexpr device_span(pointer ptr, size_type count) noexcept
		: data_(ptr)
		, size_(count)
	{}

	constexpr device_span(const device_unique_ptr<std::remove_const_t<T>> &ptr) noexcept
		requires std::is_const_v<T>
		: data_(ptr.get())
		, size_(ptr.size())
	{}

	constexpr device_span(device_unique_ptr<std::remove_const_t<T>> &ptr) noexcept
		requires (!std::is_const_v<T>)
		: data_(ptr.data_)
		, size_(ptr.size_)
	{}

	[[nodiscard]] __device__ constexpr auto data() const noexcept -> pointer { return data_; }

	[[nodiscard]]            constexpr auto size() const noexcept -> size_type { return size_; }

	[[nodiscard]]            constexpr auto size_bytes() const noexcept -> size_type { return size_ * sizeof(T); }

	[[nodiscard]]            constexpr auto empty() const noexcept -> bool { return size_ == 0; }

	[[nodiscard]] __device__ constexpr auto begin() const noexcept -> iterator { return data_; }

	[[nodiscard]] __device__ constexpr auto end() const noexcept -> iterator { return data_ + size_; }

	[[nodiscard]] __device__ constexpr auto cbegin() const noexcept -> const_iterator { return data_; }

	[[nodiscard]] __device__ constexpr auto cend() const noexcept -> const_iterator { return data_ + size_; }

	[[nodiscard]] __device__ constexpr auto operator[](size_t i) -> T & { return this->data_[i]; }

	[[nodiscard]] __device__ constexpr auto operator[](size_t i) const -> T const & { return this->data_[i]; }

	[[nodiscard]] constexpr auto subspan(
		size_type offset,
		size_type count
	) const noexcept -> device_span<T> {
		return device_span<T>(data_ + offset, count);
	}

	[[nodiscard]] constexpr auto first(size_type count) const noexcept -> device_span<T> {
		return device_span<T>(data_, count);
	}

	[[nodiscard]] constexpr auto last(size_type count) const noexcept -> device_span<T> {
		return device_span<T>(data_ + (size_ - count), count);
	}

	template <typename U>
		requires std::is_trivially_copyable_v<U>
	friend auto copy_to_device(
		const std::span<U> host,
		const device_span<U> device
	) -> void;

	template <typename U>
		requires std::is_trivially_copyable_v<U>
	friend auto copy_to_host(
		const std::span<U> host,
		const device_span<U> device
	) -> void;

	template <typename U>
		requires std::is_trivially_copyable_v<U>
	friend auto device_memset(device_span<U> device, int val) -> void;

	friend class device_unique_ptr<T>;
};

template <typename T>
	requires std::is_trivially_copyable_v<T>
class device_unique_ptr : private device_span<T> {
public:
	device_unique_ptr() noexcept = default;

	explicit device_unique_ptr(size_t n)
		: device_span<T>(nullptr, n)
	{
		if (n <= 0) {
			return;
		}

		hipError_t err = hipMalloc(&this->data_, n * sizeof(T));
		if (err != hipSuccess) {
			throw std::runtime_error("hipMalloc failed");
		}
	}

	device_unique_ptr(const device_unique_ptr &)                     = delete;
	auto operator=(const device_unique_ptr &) -> device_unique_ptr & = delete;

	device_unique_ptr(device_unique_ptr &&other) noexcept
		: device_span<T>(other.data_, other.size_)
	{
		other.data_ = nullptr;
		other.size_ = 0;
	}

	auto operator=(device_unique_ptr &&other) noexcept -> device_unique_ptr & {
		if (this != &other) {
			this->free();
			this->data_ = other.data_;
			this->size_ = other.size_;
			other.data_ = nullptr;
			other.size_ = 0;
		}
		return *this;
	}

	~device_unique_ptr() {
		free();
	}

private:
	auto free() -> void {
		if (!this->data_) {
			return;
		}
		auto err = hipFree(this->data_);
		if (err != hipSuccess) {
			throw std::runtime_error("hipFree failed");
		}
		this->data_ = nullptr;
		this->size_ = 0;
	}

	friend class device_span<T>;
};

template <typename T>
	requires std::is_trivially_copyable_v<T>
auto copy_to_device(
	const std::span<T> host,
	const device_span<T> device
) -> void {
	T *host_   = host.data();
	T *device_ = device.data_;
	auto err   = hipMemcpy(device_, host_, host.size_bytes(), hipMemcpyHostToDevice);
	if (err != hipSuccess) {
		throw new std::runtime_error("hipMemcpy (to device) failed");
	}
}

template <typename T>
	requires std::is_trivially_copyable_v<T>
auto copy_to_host(
	const std::span<T> host,
	const device_span<T> device
) -> void {
	T *host_   = host.data();
	T *device_ = device.data_;
	auto err   = hipMemcpy(host_, device_, host.size_bytes(), hipMemcpyDeviceToHost);
	if (err != hipSuccess) {
		throw new std::runtime_error("hipMemcpy (to host) failed");
	}
}

template <typename T>
	requires std::is_trivially_copyable_v<T>
auto device_memset(device_span<T> device, int val) -> void {
	auto err = hipMemset(device.data_, val, device.size_ * sizeof(T));
	if (err != hipSuccess) {
		throw new std::runtime_error("hipMemset failed");
	}
}

auto synchronise() -> void {
	auto err = hipDeviceSynchronize();
	if (err != hipSuccess) {
		throw new std::runtime_error("hipDeviceSynchronize failed");
	}
}

}
