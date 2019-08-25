package nd

import "math"

func idist(n int, x, y []float64, step int) (s float64) {
	for i := 0; n != 0; i += step {
		d := x[i] - y[i]
		s += d * d
		n--
	}
	return
}

func udist(n int, x, y []float64) (s float64) {
	_ = y[len(x[:n])-1]
	for i, xv := range x[:n] {
		d := xv - y[i]
		s += d * d
	}
	return
}

func dist2d(shape, strides Shape, x, y []float64) (s float64) {
	n := shape[1]
	step0, step1 := strides[0], strides[1]
	if step1 > 1 {
		for i := 0; i < shape[0]; i++ {
			b := step0 * i
			s += idist(n, x[b:], y[b:], step1)
		}
		return
	}

	for i := 0; i < shape[0]; i++ {
		b := step0 * i
		s += udist(n, x[b:], y[b:])
	}

	return
}

// Dist computes the euclidean distance between two arrays, x and y.
func Dist(x, y Array) (s float64) {
	if !isShapeSame(x, y) {
		panic("Dist: input arrays must have same shape")
	}

	ndims := x.Ndims()
	ashape := x.Shape()
	astrides := x.Strides()
	xd := x.Data()
	yd := y.Data()

	if ndims < 3 {
		return math.Sqrt(dist2d(ashape, astrides, xd, yd))
	}

	shape := make(Shape, ndims)
	copy(shape, ashape[:ndims-2])
	for i := ndims - 2; i < ndims; i++ {
		shape[i] = 1
	}

	ind := make(Index, ndims-2)
	shape2d := ashape[ndims-2:]
	strides2d := astrides[ndims-2:]
	istrides := ComputeStrides(shape)

	for i := 0; i < computeSize(shape); i++ {
		b := sub2ind(astrides[:ndims-2], ind2sub(istrides[:ndims-2], i, ind))
		s += dist2d(shape2d, strides2d, xd[b:], yd[b:])
	}

	return math.Sqrt(s)
}
