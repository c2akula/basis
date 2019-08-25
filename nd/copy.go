package nd

func icopy(n int, dst, src []float64, step int) {
	for i := 0; n != 0; i += step {
		dst[i] = src[i]
		n--
	}
}

func ucopy(n int, dst, src []float64) {
	_ = dst[len(src[:n])-1]
	for i, s := range src[:n] {
		dst[i] = s
	}
}

func copy2d(shape, strides Shape, dst, src []float64) {
	n := shape[1]
	step0 := strides[0]
	step1 := strides[1]

	if step1 > 1 {
		for i := 0; i < shape[0]; i++ {
			b := step0 * i
			icopy(n, dst[b:], src[b:], step1)
		}
		return
	}

	for i := 0; i < shape[0]; i++ {
		b := step0 * i
		ucopy(n, dst[b:], src[b:])
	}
}

// Copy copies src into dst.
// Note: dst and src must be of the same length.
func Copy(dst, src Array) Array {
	ndims := dst.Ndims()
	dshape := dst.Shape()
	dstrides := dst.Strides()
	dd := dst.Data()
	sd := src.Data()

	if ndims < 3 {
		copy2d(dshape, dstrides, dd, sd)
		return dst
	}

	shape := make(Shape, ndims)
	copy(shape, dshape[:ndims-2])
	for i := ndims - 2; i < ndims; i++ {
		shape[i] = 1
	}

	ind := make(Index, ndims-2)

	shape2d := dshape[ndims-2:]
	strides2d := dstrides[ndims-2:]
	istrides := ComputeStrides(shape)
	for i := 0; i < computeSize(shape); i++ {
		b := sub2ind(dstrides[:ndims-2], ind2sub(istrides[:ndims-2], i, ind))
		copy2d(shape2d, strides2d, dd[b:], sd[b:])
	}
	return dst
}
