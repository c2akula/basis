package nd

import (
	crand "crypto/rand"
	"encoding/binary"
	"math"
	"math/rand"
)

func init() {
	buf := [8]byte{}
	if _, err := crand.Read(buf[:]); err != nil {
		panic(err)
	}
	rand.Seed(int64(binary.LittleEndian.Uint64(buf[:])))
}

// New returns an array of dimensions n
// filled with elements given in v.
func New(n Shape, v []float64) Array {
	res := Zeros(n).(*ndarray)
	copy(res.data, v)
	return res
}

// Zeros returns an ndarray of type dtype and dimensions n.
func Zeros(shape Shape) Array {
	res := &ndarray{
		ndims:   len(shape),
		shape:   shape,
		strides: ComputeStrides(shape),
		size:    ComputeSize(shape),
	}

	// initialize the iterator
	res.it = newiter(res)
	res.data = make([]float64, res.size)

	return res
}

// Ones creates an array of shape n with all 1's.
func Ones(n Shape) Array {
	res := Zeros(n)
	rd := res.Data()
	for i := range rd {
		rd[i] = 1
	}
	return res
}

// Zeroslike creates an array with the shape of a but filled
// with 0's.
func Zeroslike(a Array) Array {
	return Zeros(a.Shape())
}

// Rand returns a single uniformly distributed random number in the interval [0,1)
// If n is provided then an Array of the dimensions specified is returned filled
// with random numbers.
func Rand(n Shape) Array {
	arr := Zeros(n)
	data := arr.Data()
	for i := range data {
		data[i] = rand.Float64()
	}
	return arr
}

// RandBool returns an array with uniformly distributed
// 1's and 0's
func RandBool(n Shape) Array {
	res := Zeros(n)
	data := res.Data()
	for i := range data {
		if rand.Float64() > 0.5 {
			data[i] = 1
		} else {
			data[i] = 0
		}
	}
	return res
}

// View extracts an array with dimensions given by shape
// and starting at the coordinate given by start.
// Note: Changes made to the returned array will be
// visible in the original.
func (array *ndarray) View(start Index, shape Shape) View {
	arr := &ndarray{
		data:    array.data[Sub2ind(array.strides, start):],
		ndims:   len(shape),
		shape:   shape,
		strides: array.strides[array.ndims-len(shape):],
		size:    ComputeSize(shape),
	}
	arr.it = newiter(arr)
	return arr
}

// Arange returns a array of evenly spaced values between [start, stop).
// If step is not provided, then it is set to 1, and the no. of elements
// in the array is equal to m, where m = (stop-start).
// If step is provided, then the no. of elements in the array is equal
// to m, where m = (stop-start)/step + (stop-start)%step.
func Arange(start, stop float64, step ...float64) Array {
	inc := 1.0
	m := int(stop - start)
	if len(step) > 0 {
		inc = step[0]
		q := (stop - start) / inc
		r := math.Remainder(stop-start, inc)
		m = int(q + r)
	}

	res := Zeros(Shape{1, m}).(*ndarray)
	res.data[0] = start
	for i := 1; i < res.size; i++ {
		res.data[i] = res.data[i-1] + inc
	}
	return res
}

// Reshape copies the given array into a new Array with the new shape.
// The shape should be given such that the no. of elements remains
// the same as the original.
func Reshape(array Array, shape Shape) Array {
	if array.Size() != ComputeSize(shape) {
		panic("Reshape: new shape should compute to the same no. of elements as the original")
	}

	return New(shape, array.Data())
}

// Helpers

// ComputeEnd places the cartesian coordinate index of the last element in end.
func ComputeEnd(shape Shape, end Index) {
	_ = end[len(shape)-1]
	for i, n := range shape {
		end[i] = n - 1
	}
}

// IsShapeSame checks if the two arrays have the same rank and dimensions.
func IsShapeSame(a, b Array) bool {
	if a.Ndims() != b.Ndims() {
		return false
	}

	ashape := a.Shape()
	bshape := b.Shape()

	for i := 0; i < a.Ndims(); i++ {
		if ashape[i] != bshape[i] {
			return false
		}
	}

	return true
}

// ComputeSize computes the product of the dimensions in shape.
func ComputeSize(shape Shape) int {
	size := shape[0]
	for _, n := range shape[1:] {
		size *= n
	}
	return size
}
