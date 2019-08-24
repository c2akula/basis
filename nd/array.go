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
// filled with elements given in v
func New(n Shape, v []float64) Array {
	res := Zeros(n).(*Ndarray)
	copy(res.data, v)
	return res
}

// Zeros returns an ndarray of type dtype and dimensions n
func Zeros(n Shape) Array {
	res := &Ndarray{
		ndims: len(n),
		shape: make(Shape, len(n)),
		beg:   make(Index, len(n)),
		end:   make(Index, len(n)),
	}
	// compute end coordinate of the array
	for j, i := range n {
		res.end[j] = i - 1
	}
	// copy the actual shape
	copy(res.shape, n)
	res.strides = ComputeStrides(res.shape)
	res.size = computeSize(res.shape)

	// initialize the iterator
	res.it = Iter(res)
	res.data = make([]float64, res.size)

	return res
}

// Ones creates an array of shape n with all 1's.
func Ones(n Shape) Array {
	res := Zeros(n)
	for it := res.Take(); !it.Done(); it.Next() {
		*it.Upk() = 1
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
// visible in the original. View does not initialize
// the internal iterator. Meaning, an iterator will have
// to created to iterate on the View.
// Eg:
//  m := array.View(start, shape)
//  it := Iter(m) // create an iterator to iterate through m
func (res *Ndarray) View(start Index, shape Shape) Array {
	arr := &Ndarray{
		data:    res.data[sub2ind(res.strides, start):],
		ndims:   len(shape),
		shape:   shape,
		strides: res.strides[res.ndims-len(shape):],
		size:    computeSize(shape),
	}
	be := make(Index, arr.ndims*2)
	arr.beg, arr.end = be[:arr.ndims], be[arr.ndims:]
	computeEnd(arr.shape, arr.end)
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

	res := Zeros(Shape{1, m}).(*Ndarray)
	res.data[0] = start
	for i := 1; i < res.size; i++ {
		res.data[i] = res.data[i-1] + inc
	}
	return res
}

// Reshape changes the dimensions of the array to shape specified.
// The shape should be given such that the no. of elements remains
// the same as the original.
func Reshape(array Array, shape Shape) Array {
	arr := array.(*Ndarray)
	if arr.size != computeSize(shape) {
		panic("new shape should compute to the same no. of elements as the original")
	}
	arr.shape = make(Shape, len(shape))
	copy(arr.shape, shape)

	arr.strides = ComputeStrides(arr.shape)
	arr.ndims = len(arr.shape)

	arr.beg = make(Index, arr.ndims)
	arr.end = make(Index, arr.ndims)
	computeEnd(arr.shape, arr.end)
	arr.it = Iter(arr)
	return arr
}

// Helpers

func computeEnd(shape Shape, end Index) {
	_ = end[len(shape)-1]
	for i, n := range shape {
		end[i] = n - 1
	}
}

func isShapeSame(a, b Array) bool {
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

func (res *Ndarray) computeStrides() {
	for k := 0; k < res.ndims; k++ {
		res.strides[k] = 1
		for l := k + 1; l < res.ndims; l++ {
			res.strides[k] *= res.shape[l]
		}
	}
}

func computeSize(shape Shape) int {
	size := shape[0]
	for _, n := range shape[1:] {
		size *= n
	}
	return size
}
