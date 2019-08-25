package nd

func idot(n int, x, y []float64, step int) (s float64) {
	for i := 0; n != 0; i += step {
		s += x[i] * y[i]
		n--
	}
	return
}

func udot(n int, x, y []float64) (s float64) {
	_ = y[len(x[:n])-1]
	for i, xv := range x[:n] {
		s += xv * y[i]
	}
	// for i := 0; n != 0; i++ {
	// 	s += x[i] * y[i]
	// 	n--
	// }
	return
}

func dot2d(shape, xstrides Shape, x, y []float64) (s float64) {
	n := shape[1]
	step0 := xstrides[0]
	step1 := xstrides[1]
	if step1 > 1 {
		for i := 0; i < shape[0]; i++ {
			b := step0 * i
			s += idot(n, x[b:], y[b:], step1)
		}
		return
	}

	for i := 0; i < shape[0]; i++ {
		b := step0 * i
		s += udot(n, x[b:], y[b:])
	}

	return s
}

// Dot computes the inner product of two arrays, x and y.
// Note: x and y must have the same shape.
func Dot(x, y Array) (s float64) {
	if !isShapeSame(x, y) {
		panic("Dot: input arrays must have same shape")
	}

	ndims := x.Ndims()
	xshape := x.Shape()
	xstrides := x.Strides()
	xd := x.Data()
	yd := y.Data()

	if ndims < 3 {
		return dot2d(xshape, xstrides, xd, yd)
	}

	shape := make(Shape, ndims)
	copy(shape, xshape[:ndims-2])
	for i := ndims - 2; i < ndims; i++ {
		shape[i] = 1
	}

	ind := make(Index, ndims-2)

	shape2d := xshape[ndims-2:]
	strides2d := xstrides[ndims-2:]
	istrides := ComputeStrides(shape)
	for i := 0; i < computeSize(shape); i++ {
		b := sub2ind(xstrides[:ndims-2], ind2sub(istrides[:ndims-2], i, ind))
		s += dot2d(shape2d, strides2d, xd[b:], yd[b:])
	}
	return
}
