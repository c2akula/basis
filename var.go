package go_nd

import (
	"math"

	"github.com/c2akula/go.nd/nd"
)

func ivar(n int, mu float64, x []float64, step int) (s float64) {
	for i := 0; n != 0; i += step {
		v := x[i] - mu
		s += v * v
	}
	return
}

func uvar(n int, mu float64, x []float64) (s float64) {
	for _, v := range x[:n] {
		e := v - mu
		s += e * e
	}
	return
}

func var2d(shape, strides nd.Shape, mu float64, x []float64) (s float64) {
	n := shape[1]
	step0, step1 := strides[0], strides[1]
	if step1 > 1 {
		for i := 0; i < shape[0]; i++ {
			b := step0 * i
			s += ivar(n, mu, x[b:], step1)
		}
		return
	}

	for i := 0; i < shape[0]; i++ {
		b := step0 * i
		s += uvar(n, mu, x[b:])
	}
	return
}

// Var computes the variance of the array.
func Var(x nd.Array) (s float64) {
	// mu := Mean(x)
	ndims := x.Ndims()
	xshape := x.Shape()
	xstrides := x.Strides()
	xd := x.Data()

	if ndims < 3 {
		mu := sum2d(xshape, xstrides, xd) / float64(x.Size())
		return var2d(xshape, xstrides, mu, xd) / float64(x.Size()-1)
	}

	shape := make(nd.Shape, ndims)
	copy(shape, xshape[:ndims-2])
	for i := ndims - 2; i < ndims; i++ {
		shape[i] = 1
	}
	ind := make(nd.Index, ndims-2)
	shape2d := xshape[ndims-2:]
	strides2d := xstrides[ndims-2:]
	istrides := nd.ComputeStrides(shape)

	b := make(nd.Index, nd.ComputeSize(shape))
	mu := 0.0
	for i := range b {
		b[i] = nd.Sub2ind(xstrides[:ndims-2], nd.Ind2sub(istrides[:ndims-2], i, ind))
		mu += sum2d(shape2d, strides2d, xd[b[i]:]) / float64(x.Size())
	}

	for _, k := range b {
		s += var2d(shape2d, strides2d, mu, xd[k:])
	}
	return s / float64(x.Size()-1)
}

// Std computes the standard deviation of the array.
func Std(x nd.Array) (s float64) { return math.Sqrt(Var(x)) }
