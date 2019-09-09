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
func New(n Shape, v []float64) *Ndarray {
	res := Zeros(n)
	copy(res.data, v)
	return res
}

// Zeros returns an ndarray of type dtype and dimensions n.
func Zeros(shape Shape) *Ndarray {
	res := &Ndarray{
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
func Ones(n Shape) *Ndarray {
	res := Zeros(n)
	rd := res.Data()
	for i := range rd {
		rd[i] = 1
	}
	return res
}

// Zeroslike creates an array with the shape of a but filled
// with 0's.
func Zeroslike(a *Ndarray) *Ndarray {
	return Zeros(a.Shape())
}

// Rand returns a single uniformly distributed random number in the interval [0,1)
// If n is provided then an *Ndarray of the dimensions specified is returned filled
// with random numbers.
func Rand(n Shape) *Ndarray {
	arr := Zeros(n)
	data := arr.Data()
	for i := range data {
		data[i] = rand.Float64()
	}
	return arr
}

// RandBool returns an array with uniformly distributed
// 1's and 0's
func RandBool(n Shape) *Ndarray {
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
func (array *Ndarray) View(start Index, shape Shape) *Ndarray {
	arr := &Ndarray{
		data:  array.data[array.sub2ind(start):],
		ndims: len(shape),
		size:  ComputeSize(shape),
	}
	arr.shape = make(Shape, arr.ndims)
	copy(arr.shape, shape)
	arr.strides = make(Shape, arr.ndims)
	copy(arr.strides, array.strides[array.ndims-arr.ndims:])
	arr.it = newiter(arr)
	return arr
}

// Reshape changes array's shape to the new shape, shp.
func (array *Ndarray) Reshape(shp Shape) *Ndarray {
	if ComputeSize(shp) != array.size {
		panic("new shape should compute to the same size as the original")
	}

	if array.isView() {
		res := Zeros(shp)
		rit := NewNditer(res)
		ait := NewNditer(array)
		for k := 0; k < rit.sz; k++ {
			rv, rn := rit.At(k)
			av, _ := ait.At(k)
			for i, j := 0, 0; rn != 0; i, j = i+rit.rs, j+ait.rs {
				rv[i] = av[j]
				rn--
			}
		}
		// for inc := rit.Stride(); rit.Next(); {
		// 	rv, n := rit.Get()
		// 	fmt.Println("rv: ", rv, n, rit.b, rit.k)
		// 	av, n := ait.Get()
		// 	k := 0
		// 	for j := 0; n != 0; j += inc {
		// 		_ = rv[j]
		// 		_ = av[k]
		// 		n--
		// 		k += ait.Stride()
		// 	}
		// }
		// rit.Reset()
		// ait.Reset()
		return res
	}

	res := array.View(make(Index, array.ndims), array.shape)
	res.ndims = len(shp)
	if d := res.ndims - cap(res.shape); d > 0 {
		res.shape = append(res.shape, make(Shape, d)...)
		res.strides = append(res.strides, make(Shape, d)...)
	} else {
		res.shape = res.shape[:res.ndims]
		res.strides = res.strides[:res.ndims]
	}

	copy(res.shape, shp)
	computestrides(res.shape, res.strides)
	return res
}

// Permute reorders the dimensions of the array in the specified
// order. If order is nil, the order of the dimensions is reversed.
func (array *Ndarray) Permute(order []int) *Ndarray {
	res := array.View(make(Index, array.ndims), array.shape)

	if order == nil {
		order = make([]int, res.ndims)
		for k, j := res.ndims-1, 0; k >= 0; k, j = k-1, j+1 {
			order[j] = k
		}
	} else if len(order) != res.ndims {
		panic("no. of dimensions in the permutation order list != array.ndims")
	}
	/*
		https://stackoverflow.com/a/7366196
		for i in xrange(len(a)):
			x = a[i]
			j = i
			while True:
				k = indices[j]
				indices[j] = j
				if k == i:
					break
				a[j] = a[k]
				j = k
			a[j] = x
	*/

	shp := res.shape
	str := res.strides
	for k := 0; k < res.ndims; k++ {
		xshp := shp[k]
		xstr := str[k]
		j := k
		for {
			i := order[j]
			order[j] = j
			if i == k {
				break
			}
			shp[j] = shp[i]
			str[j] = str[i]
			j = i
		}
		shp[j] = xshp
		str[j] = xstr
	}
	return res
	// return array
}

// Arange returns a array of evenly spaced values between [start, stop).
// If step is not provided, then it is set to 1, and the no. of elements
// in the array is equal to m, where m = (stop-start).
// If step is provided, then the no. of elements in the array is equal
// to m, where m = (stop-start)/step + (stop-start)%step.
func Arange(start, stop float64, step ...float64) *Ndarray {
	inc := 1.0
	m := int(stop - start)
	if len(step) > 0 {
		inc = step[0]
		q := (stop - start) / inc
		r := math.Remainder(stop-start, inc)
		m = int(q + r)
	}

	res := Zeros(Shape{1, m})
	res.data[0] = start
	for i := 1; i < res.size; i++ {
		res.data[i] = res.data[i-1] + inc
	}
	return res
}

// Reshape copies the given array into a new Array with the new shape.
// The shape should be given such that the no. of elements remains
// the same as the original.
func Reshape(array Array, shape Shape) *Ndarray {
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
