#pragma once

#include "gpu.hpp"

namespace gpu {

template <typename T>
	requires std::is_trivially_copyable_v<T>
class device_unique_ptr2;

template <typename T>
	requires std::is_trivially_copyable_v<T>
class device_span2 : public device_span<T> {
public:
	using base = device_span<T>;
	using typename base::size_type;
	using typename base::pointer;

protected:
	size_type width_  = 0;
	size_type height_ = 0;
	size_type pitch_  = 0;

public:
	constexpr device_span2() noexcept = default;

	constexpr device_span2(
		pointer ptr,
		size_type width,
		size_type height,
		size_type pitch
	) noexcept
		: base(ptr, pitch * height)
		, width_(width)
		, height_(height)
		, pitch_(pitch)
	{}

	constexpr device_span2(const device_unique_ptr2<std::remove_const_t<T>> &ptr) noexcept
		requires std::is_const_v<T>
		: base(ptr.data_, ptr.pitch_ * ptr.height_)
		, width_(ptr.width_)
		, height_(ptr.height_)
		, pitch_(ptr.pitch_)
	{}

	constexpr device_span2(device_unique_ptr2<std::remove_const_t<T>> &ptr) noexcept
		requires (!std::is_const_v<T>)
		: base(ptr.data_, ptr.pitch_ * ptr.height_)
		, width_(ptr.width_)
		, height_(ptr.height_)
		, pitch_(ptr.pitch_)
	{}

	[[nodiscard]] constexpr auto width() const noexcept -> size_type { return width_; }
	[[nodiscard]] constexpr auto height() const noexcept -> size_type { return height_; }
	[[nodiscard]] constexpr auto pitch() const noexcept -> size_type { return pitch_; }

	[[nodiscard]] __device__ constexpr auto operator[](size_type y, size_type x) -> T& {
		return this->data_[get_index(y, x)];
	}

	[[nodiscard]] __device__ constexpr auto operator[](size_type y, size_type x) const -> const T& {
		return this->data_[get_index(y, x)];
	}

	[[nodiscard]] __device__ constexpr auto operator()(size_type y, size_type x) -> T& {
		return (*this)[y, x];
	}

	[[nodiscard]] __device__ constexpr auto operator()(size_type y, size_type x) const -> const T& {
		return (*this)[y, x];
	}

	[[nodiscard]] __device__ constexpr auto operator[](size_t i) -> T & { return this->data_[i]; }

	[[nodiscard]] __device__ constexpr auto operator[](size_t i) const -> T const & { return this->data_[i]; }

	[[nodiscard]] constexpr auto in_bounds(size_t i, size_t j) const -> bool { return i < width_ && j < height_; }

	[[nodiscard]] __host__ __device__ constexpr auto get_index(size_type y, size_type x) const -> size_type {
#ifndef __HIP_DEVICE_COMPILE__
		if (x >= width_ || y >= height_) {
			throw std::out_of_range("Index out of bounds in get_index");
		}
#endif
		return y * pitch_ + x;
	}

	[[nodiscard]] constexpr auto row(size_type y) const -> device_span<T> {
#ifndef __HIP_DEVICE_COMPILE__
		if (y >= height_) {
			throw std::out_of_range("Row index out of bounds");
		}
#endif
		return device_span<T>(this->data_ + y * pitch_, width_);
	}

	[[nodiscard]] __host__ __device__ constexpr auto subspan2(
		size_type x_offset,
		size_type y_offset,
		size_type width,
		size_type height
	) const -> device_span2<T> {
#ifndef __HIP_DEVICE_COMPILE__
		if ((x_offset + width) > width_ || (y_offset + height) > height_) {
			throw std::out_of_range("Out of bounds subspan");
		}
#endif

		return device_span2<T>(
			this->data_ + y_offset * pitch_ + x_offset,
			width,
			height,
			pitch_
		);
	}

	template <typename U>
		requires std::is_trivially_copyable_v<U>
	friend auto copy_to_device(
		const std::span<U> host,
		const device_span2<U> device
	) -> void;

	template <typename U>
		requires std::is_trivially_copyable_v<U>
	friend auto copy_to_host(
		const std::span<U> host,
		const device_span2<U> device
	) -> void;

	template <typename U>
		requires std::is_trivially_copyable_v<U>
	friend auto device_memset(device_span2<U> device, int val) -> void;

	friend class device_unique_ptr2<T>;
};

template <typename T>
	requires std::is_trivially_copyable_v<T>
class device_unique_ptr2 : public device_span2<T> {
public:
	using base = device_span2<T>;
	using typename base::size_type;
	using typename base::pointer;

	device_unique_ptr2() noexcept = default;

	explicit device_unique_ptr2(size_type width, size_type height)
		: device_span2<T>(nullptr, width, height, 0)
	{
		if (width <= 0 || height <= 0) {
			throw std::invalid_argument("Width and height must be positive");
		}

		size_t pitch_bytes = 0;
		hipError_t err = hipMallocPitch(
			reinterpret_cast<void**>(&this->data_),
			&pitch_bytes,
			width * sizeof(T),
			height
		);

		if (err != hipSuccess) {
			throw std::runtime_error("hipMallocPitch failed");
		}

		this->pitch_ = pitch_bytes / sizeof(T);
		this->size_ = this->height_ * this->pitch_;
	}

	device_unique_ptr2(const device_unique_ptr2 &) = delete;
	auto operator=(const device_unique_ptr2 &) -> device_unique_ptr2 & = delete;

	device_unique_ptr2(device_unique_ptr2 &&other) noexcept
		: device_span2<T>(other.data_, other.width_, other.height_, other.pitch_)
	{
		other.data_ = nullptr;
		other.width_ = 0;
		other.height_ = 0;
		other.pitch_ = 0;
		other.size_ = 0;
	}

	auto operator=(device_unique_ptr2 &&other) noexcept -> device_unique_ptr2 & {
		if (this != &other) {
			this->free();
			this->data_ = other.data_;
			this->width_ = other.width_;
			this->height_ = other.height_;
			this->pitch_ = other.pitch_;
			this->size_ = other.size_;
			other.data_ = nullptr;
			other.width_ = 0;
			other.height_ = 0;
			other.pitch_ = 0;
			other.size_ = 0;
		}
		return *this;
	}

	~device_unique_ptr2() {
		free();
	}

	operator device_span2<T>() const noexcept {
		return device_span2<T>(this->data_, this->width_, this->height_, this->pitch_);
	}

	[[nodiscard]] constexpr auto width() const noexcept -> size_type { return this->width_; }
	[[nodiscard]] constexpr auto height() const noexcept -> size_type { return this->height_; }
	[[nodiscard]] constexpr auto pitch() const noexcept -> size_type { return this->pitch_; }

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
		this->width_ = 0;
		this->height_ = 0;
		this->pitch_ = 0;
		this->size_ = 0;
	}

	friend class device_span2<T>;
};

template <typename U>
	requires std::is_trivially_copyable_v<U>
auto copy_to_device(
	const std::span<U> host,
	const device_span2<U> device
) -> void {
	if (host.size_bytes() != device.size_bytes()) {
		throw std::invalid_argument("hipMemcpy2D (host to device) failed, differing sizes");
	}

	const U *host_ptr = host.data();
	U *device_ptr = device.data_;

	auto err = hipMemcpy2D(
		device_ptr,
		device.pitch_ * sizeof(U),
		host_ptr,
		device.width_ * sizeof(U),
		device.width_ * sizeof(U),
		device.height_,
		hipMemcpyHostToDevice
	);

	if (err != hipSuccess) {
		throw std::runtime_error("hipMemcpy2D (to device) failed");
	}
}

template <typename U>
	requires std::is_trivially_copyable_v<U>
auto copy_to_host(
	const std::span<U> host,
	const device_span2<U> device
) -> void {
	if (host.size_bytes() != device.size_bytes()) {
		throw std::invalid_argument("hipMemcpy2D (host to device) failed, differing sizes");
	}

	U *host_ptr = host.data();
	const U *device_ptr = device.data_;

	auto err = hipMemcpy2D(
		host_ptr,
		device.width_ * sizeof(U),
		device_ptr,
		device.pitch_ * sizeof(U),
		device.width_ * sizeof(U),
		device.height_,
		hipMemcpyDeviceToHost
	);

	if (err != hipSuccess) {
		throw std::runtime_error("hipMemcpy2D (to host) failed");
	}
}

template <typename U>
	requires std::is_trivially_copyable_v<U>
auto device_memset(device_span2<U> device, int val) -> void {
	auto err = hipMemset2D(
		device.data_,
		device.pitch_ * sizeof(U),
		val,
		device.width_ * sizeof(U),
		device.height_
	);

	if (err != hipSuccess) {
		throw std::runtime_error("hipMemset2D failed");
	}
}

}
