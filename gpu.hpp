#pragma once

#include "runtime.hpp"

#define TEMPLATE_COPYABLE(T) template <typename T> requires std::is_trivially_copyable_v<T>

namespace gpu {

TEMPLATE_COPYABLE(T)
class device_unique_ptr;

TEMPLATE_COPYABLE(T)
class device_span {
public:
	using element_type   = T;
	using value_type     = std::remove_cv_t<T>;
	using size_type      = std::size_t;
	using pointer        = std::conditional_t<
		std::is_const_v<T>,
		T * __restrict__,
		T *
	>;
	using const_pointer  = const T * __restrict__;
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
		: data_(ptr.data_)
		, size_(ptr.size_)
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

	[[nodiscard]]            constexpr auto in_bounds(size_t i) const -> bool { return i < size_; }

	[[nodiscard]] __host__ __device__ constexpr auto subspan(
		size_type offset,
		size_type count
	) const -> device_span<T> {
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
		if ((offset + count) > size_) {
			throw std::out_of_range("Out of bounds subspan");
		}
#endif
		return device_span<T>(data_ + offset, count);
	}

	[[nodiscard]] constexpr auto first(size_type count) const -> device_span<T> {
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
		if (count > size_) {
			throw std::out_of_range("Count exceeds span size in first");
		}
#endif
		return device_span<T>(data_, count);
	}

	[[nodiscard]] constexpr auto last(size_type count) const -> device_span<T> {
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
		if (count > size_) {
			throw std::out_of_range("Count exceeds span size in last");
		}
#endif
		return device_span<T>(data_ + (size_ - count), count);
	}

	TEMPLATE_COPYABLE(U)
	friend auto memcpy(
		const device_span<U> dest,
		const device_span<U> src
	) -> void;

	TEMPLATE_COPYABLE(U)
	friend auto copy_to_device(
		const std::span<U> host,
		const device_span<U> device
	) -> void;

	TEMPLATE_COPYABLE(U)
	friend auto copy_to_host(
		const std::span<U> host,
		const device_span<U> device
	) -> void;

	TEMPLATE_COPYABLE(U)
	friend auto device_memset(device_span<U> device, int val) -> void;

	friend class device_unique_ptr<T>;
};

template <typename T>
	requires std::is_trivially_copyable_v<T>
class device_unique_ptr : public device_span<T> {
public:
	device_unique_ptr() noexcept = default;

	explicit device_unique_ptr(size_t n)
		: device_span<T>(nullptr, n)
	{
		if (n <= 0) {
			throw std::invalid_argument("Size must be positive");
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

TEMPLATE_COPYABLE(T)
auto memcpy(
	const device_span<T> dest,
	const device_span<T> src
) -> void {
	if (dest.size_bytes() != src.size_bytes()) {
		throw std::invalid_argument("hipMemcpy (device to device) failed, differing sizes");
	}
	T *from = src.data_;
	T *to = dest.data_;
	auto err = hipMemcpy(to, from, dest.size_bytes(), hipMemcpyDeviceToDevice);
	if (err != hipSuccess) {
		throw std::runtime_error("hipMemcpy (device to device) failed");
	}
}

TEMPLATE_COPYABLE(T)
auto copy_to_device(
	const std::span<T> host,
	const device_span<T> device
) -> void {
	if (host.size_bytes() != device.size_bytes()) {
		throw std::invalid_argument("hipMemcpy (to device) failed, differing sizes");
	}
	T *host_   = host.data();
	T *device_ = device.data_;
	auto err   = hipMemcpy(device_, host_, host.size_bytes(), hipMemcpyHostToDevice);
	if (err != hipSuccess) {
		throw std::runtime_error("hipMemcpy (to device) failed");
	}
}

TEMPLATE_COPYABLE(T)
auto copy_to_device(
	const std::span<const T> host,
	const device_span<T> device
) -> void {
	gpu::copy_to_device(std::span<T>(const_cast<T*>(host.data()), host.size()), device);
}

TEMPLATE_COPYABLE(T)
auto copy_to_host(
	const std::span<T> host,
	const device_span<T> device
) -> void {
	if (host.size_bytes() != device.size_bytes()) {
		throw std::invalid_argument("hipMemcpy (to host) failed, differing sizes");
	}
	T *host_   = host.data();
	T *device_ = device.data_;
	auto err   = hipMemcpy(host_, device_, host.size_bytes(), hipMemcpyDeviceToHost);
	if (err != hipSuccess) {
		throw std::runtime_error("hipMemcpy (to host) failed");
	}
}

TEMPLATE_COPYABLE(T)
auto copy_to_host(
	const std::span<T> host,
	const device_span<const T> device
) -> void {
	gpu::copy_to_device(host, device_span<T>{device.data_, device.size()});
}

TEMPLATE_COPYABLE(T)
auto device_memset(device_span<T> device, int val) -> void {
	auto err = hipMemset(device.data_, val, device.size_ * sizeof(T));
	if (err != hipSuccess) {
		throw std::runtime_error("hipMemset failed");
	}
}

inline auto synchronise() -> void {
	auto err = hipDeviceSynchronize();
	if (err != hipSuccess) {
		throw std::runtime_error("hipDeviceSynchronize failed");
	}
}

}
