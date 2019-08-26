package go_nd

import "github.com/c2akula/go.nd/nd"

func isum(n int, x []float64, step int) (s float64) {
	for i := 0; n != 0; i += step {
		s += x[i]
		n--
	}
	return
}

func usum(n int, x []float64) (s float64) {
	for _, v := range x[:n] {
		s += v
	}
	return
}

func sum2d(shape, strides nd.Shape, x []float64) (s float64) {
	n := shape[1]
	step0, step1 := strides[0], strides[1]
	if step1 > 1 {
		for i := 0; i < shape[0]; i++ {
			b := step0 * i
			s += isum(n, x[b:], step1)
		}
		return
	}

	for i := 0; i < shape[0]; i++ {
		b := step0 * i
		s += usum(n, x[b:])
	}
	return
}

func Sum(x nd.Array) (s float64) {
	ndims := x.Ndims()
	xshape := x.Shape()
	xstrides := x.Strides()
	xd := x.Data()

	if ndims < 3 {
		return sum2d(xshape, xstrides, xd)
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
	for i := 0; i < nd.ComputeSize(shape); i++ {
		b := nd.Sub2ind(xstrides[:ndims-2], nd.Ind2sub(istrides[:ndims-2], i, ind))
		s += sum2d(shape2d, strides2d, xd[b:])
	}
	return
}

// Mean computes the average of the elements pointed to
// by the iterator it.
func Mean(x nd.Array) (m float64) {
	return Sum(x) * (1.0 / float64(x.Size()))
}
