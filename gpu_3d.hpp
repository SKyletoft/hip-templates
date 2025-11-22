#pragma once

#include "gpu.hpp"

namespace gpu {

template <typename T>
	requires std::is_trivially_copyable_v<T>
class device_unique_ptr3;

template <typename T>
	requires std::is_trivially_copyable_v<T>
class device_span3 : public device_span<T> {
public:
	using base = device_span<T>;
	using typename base::element_type;
	using typename base::value_type;
	using typename base::size_type;
	using typename base::pointer;
	using typename base::const_pointer;

protected:
	size_type width_  = 0;
	size_type height_ = 0;
	size_type depth_  = 0;
	size_type pitch_  = 0;
	size_type slice_pitch_ = 0;

public:
	constexpr device_span3() noexcept = default;

	constexpr device_span3(
		pointer ptr,
		size_type width,
		size_type height,
		size_type depth,
		size_type pitch,
		size_type slice_pitch
	) noexcept
		: base(ptr, width * height * depth)
		, width_(width)
		, height_(height)
		, depth_(depth)
		, pitch_(pitch)
		, slice_pitch_(slice_pitch)
	{}

	constexpr device_span3(const device_unique_ptr3<std::remove_const_t<T>> &ptr) noexcept
		requires std::is_const_v<T>
		: base(ptr.data_, ptr.width_ * ptr.height_ * ptr.depth_)
		, width_(ptr.width_)
		, height_(ptr.height_)
		, depth_(ptr.depth_)
		, pitch_(ptr.pitch_)
		, slice_pitch_(ptr.slice_pitch_)
	{}

	constexpr device_span3(device_unique_ptr3<std::remove_const_t<T>> &ptr) noexcept
		requires (!std::is_const_v<T>)
		: base(ptr.data_, ptr.width_ * ptr.height_ * ptr.depth_)
		, width_(ptr.width_)
		, height_(ptr.height_)
		, depth_(ptr.depth_)
		, pitch_(ptr.pitch_)
		, slice_pitch_(ptr.slice_pitch_)
	{}

	[[nodiscard]] constexpr auto width() const noexcept -> size_type { return width_; }
	[[nodiscard]] constexpr auto height() const noexcept -> size_type { return height_; }
	[[nodiscard]] constexpr auto depth() const noexcept -> size_type { return depth_; }
	[[nodiscard]] constexpr auto pitch() const noexcept -> size_type { return pitch_; }
	[[nodiscard]] constexpr auto slice_pitch() const noexcept -> size_type { return slice_pitch_; }

	[[nodiscard]] __device__ constexpr auto operator()(size_type x, size_type y, size_type z) -> T& {
		return this->data_[get_index(x, y, z)];
	}

	[[nodiscard]] __device__ constexpr auto operator()(size_type x, size_type y, size_type z) const -> const T& {
		return this->data_[get_index(x, y, z)];
	}

	[[nodiscard]] __host__ __device__ constexpr auto get_index(size_type x, size_type y, size_type z) const -> size_type {
#ifndef __HIP_DEVICE_COMPILE__
		if (x >= width_ || y >= height_ || z >= depth_) {
			throw std::out_of_range("Index out of bounds in get_index");
		}
#endif
		return z * slice_pitch_ + y * pitch_ + x;
	}

	[[nodiscard]] constexpr auto slice(size_type z) const -> device_span2<T> {
#ifndef __HIP_DEVICE_COMPILE__
		if (z >= depth_) {
			throw std::out_of_range("Slice index out of bounds");
		}
#endif
		return device_span2<T>(
			this->data_ + z * slice_pitch_,
			width_,
			height_,
			pitch_
		);
	}

	[[nodiscard]] constexpr auto row(size_type y, size_type z) const -> device_span<T> {
#ifndef __HIP_DEVICE_COMPILE__
		if (y >= height_ || z >= depth_) {
			throw std::out_of_range("Row index out of bounds");
		}
#endif
		return device_span<T>(this->data_ + z * slice_pitch_ + y * pitch_, width_);
	}

	[[nodiscard]] __host__ __device__ constexpr auto subspan3(
		size_type x_offset,
		size_type y_offset,
		size_type z_offset,
		size_type width,
		size_type height,
		size_type depth
	) const -> device_span3<T> {
#ifndef __HIP_DEVICE_COMPILE__
		if ((x_offset + width) > width_ || (y_offset + height) > height_ || (z_offset + depth) > depth_) {
			throw std::out_of_range("Out of bounds subspan");
		}
#endif
		return device_span3<T>(
			this->data_ + z_offset * slice_pitch_ + y_offset * pitch_ + x_offset,
			width,
			height,
			depth,
			pitch_,
			slice_pitch_
		);
	}

	template <typename U>
		requires std::is_trivially_copyable_v<U>
	friend auto copy_to_device(
		const std::span<U> host,
		const device_span3<U> device
	) -> void;

	template <typename U>
		requires std::is_trivially_copyable_v<U>
	friend auto copy_to_host(
		const std::span<U> host,
		const device_span3<U> device
	) -> void;

	template <typename U>
		requires std::is_trivially_copyable_v<U>
	friend auto device_memset(device_span3<U> device, int val) -> void;

	friend class device_unique_ptr3<T>;
};

template <typename T>
	requires std::is_trivially_copyable_v<T>
class device_unique_ptr3 : public device_span3<T> {
public:
	using base = device_span3<T>;
	using typename base::size_type;
	using typename base::pointer;

	device_unique_ptr3() noexcept = default;

	explicit device_unique_ptr3(size_type width, size_type height, size_type depth)
		: device_span3<T>(nullptr, width, height, depth, 0, 0)
	{
		if (width <= 0 || height <= 0 || depth <= 0) {
			throw std::invalid_argument("Width, height, and depth must be positive");
		}

		hipExtent extent = make_hipExtent(width * sizeof(T), height, depth);
		hipPitchedPtr pitched_ptr;

		hipError_t err = hipMalloc3D(&pitched_ptr, extent);

		if (err != hipSuccess) {
			throw std::runtime_error("hipMalloc3D failed");
		}

		this->data_ = static_cast<pointer>(pitched_ptr.ptr);
		this->pitch_ = pitched_ptr.pitch / sizeof(T);
		this->slice_pitch_ = (pitched_ptr.pitch / sizeof(T)) * height;
	}

	device_unique_ptr3(const device_unique_ptr3 &) = delete;
	auto operator=(const device_unique_ptr3 &) -> device_unique_ptr3 & = delete;

	device_unique_ptr3(device_unique_ptr3 &&other) noexcept
		: device_span3<T>(other.data_, other.width_, other.height_, other.depth_, other.pitch_, other.slice_pitch_)
	{
		other.data_ = nullptr;
		other.width_ = 0;
		other.height_ = 0;
		other.depth_ = 0;
		other.pitch_ = 0;
		other.slice_pitch_ = 0;
		other.size_ = 0;
	}

	auto operator=(device_unique_ptr3 &&other) noexcept -> device_unique_ptr3 & {
		if (this != &other) {
			this->free();
			this->data_ = other.data_;
			this->width_ = other.width_;
			this->height_ = other.height_;
			this->depth_ = other.depth_;
			this->pitch_ = other.pitch_;
			this->slice_pitch_ = other.slice_pitch_;
			this->size_ = other.size_;
			other.data_ = nullptr;
			other.width_ = 0;
			other.height_ = 0;
			other.depth_ = 0;
			other.pitch_ = 0;
			other.slice_pitch_ = 0;
			other.size_ = 0;
		}
		return *this;
	}

	~device_unique_ptr3() {
		free();
	}

	operator device_span3<T>() const noexcept {
		return device_span3<T>(this->data_, this->width_, this->height_, this->depth_, this->pitch_, this->slice_pitch_);
	}

	[[nodiscard]] constexpr auto width() const noexcept -> size_type { return this->width_; }
	[[nodiscard]] constexpr auto height() const noexcept -> size_type { return this->height_; }
	[[nodiscard]] constexpr auto depth() const noexcept -> size_type { return this->depth_; }
	[[nodiscard]] constexpr auto pitch() const noexcept -> size_type { return this->pitch_; }
	[[nodiscard]] constexpr auto slice_pitch() const noexcept -> size_type { return this->slice_pitch_; }

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
		this->depth_ = 0;
		this->pitch_ = 0;
		this->slice_pitch_ = 0;
		this->size_ = 0;
	}

	friend class device_span3<T>;
};

template <typename U>
	requires std::is_trivially_copyable_v<U>
auto copy_to_device(
	const std::span<U> host,
	const device_span3<U> device
) -> void {
	if (host.size_bytes() != device.size_bytes()) {
		throw std::invalid_argument("hipMemcpy3D (to device) failed, differing sizes");
	}

	const U *host_ptr = host.data();

	hipMemcpy3DParms params = {0};
	params.srcPtr = make_hipPitchedPtr(
		const_cast<U*>(host_ptr),
		device.width_ * sizeof(U),
		device.width_,
		device.height_
	);
	params.dstPtr = make_hipPitchedPtr(
		device.data_,
		device.pitch_ * sizeof(U),
		device.width_,
		device.height_
	);
	params.extent = make_hipExtent(
		device.width_ * sizeof(U),
		device.height_,
		device.depth_
	);
	params.kind = hipMemcpyHostToDevice;

	auto err = hipMemcpy3D(&params);

	if (err != hipSuccess) {
		throw std::runtime_error("hipMemcpy3D (to device) failed");
	}
}

template <typename U>
	requires std::is_trivially_copyable_v<U>
auto copy_to_host(
	const std::span<U> host,
	const device_span3<U> device
) -> void {
	if (host.size_bytes() != device.size_bytes()) {
		throw std::invalid_argument("hipMemcpy3D (to host) failed, differing sizes");
	}

	U *host_ptr = host.data();

	hipMemcpy3DParms params = {0};
	params.srcPtr = make_hipPitchedPtr(
		device.data_,
		device.pitch_ * sizeof(U),
		device.width_,
		device.height_
	);
	params.dstPtr = make_hipPitchedPtr(
		host_ptr,
		device.width_ * sizeof(U),
		device.width_,
		device.height_
	);
	params.extent = make_hipExtent(
		device.width_ * sizeof(U),
		device.height_,
		device.depth_
	);
	params.kind = hipMemcpyDeviceToHost;

	auto err = hipMemcpy3D(&params);

	if (err != hipSuccess) {
		throw std::runtime_error("hipMemcpy3D (to host) failed");
	}
}

template <typename U>
	requires std::is_trivially_copyable_v<U>
auto device_memset(device_span3<U> device, int val) -> void {
	hipPitchedPtr pitched_ptr = make_hipPitchedPtr(
		device.data_,
		device.pitch_ * sizeof(U),
		device.width_,
		device.height_
	);

	hipExtent extent = make_hipExtent(
		device.width_ * sizeof(U),
		device.height_,
		device.depth_
	);

	auto err = hipMemset3D(pitched_ptr, val, extent);

	if (err != hipSuccess) {
		throw std::runtime_error("hipMemset3D failed");
	}
}

}
