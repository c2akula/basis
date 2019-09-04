package basis

import "github.com/c2akula/basis/nd"

// Copy copies min(dst.Len(), src.Len()) no. of elements from src into dst.
// Note: If dst and src are the same, then dst is returned unmodified.
func Copy(dst, src nd.Iterator) nd.Iterator {
	if dst == src {
		return dst
	}

	n := dst.Len()
	if src.Len() < dst.Len() {
		n = src.Len()
	}

	d := dst.Data()
	s := src.Data()

	di := dst.Ind()
	si := src.Ind()

	for i, k := range di[:n] {
		d[k] = s[si[i]]
	}

	return dst
}

func icopy(n int, dst, src []float64, step int) {
	for i := 0; n != 0; i += step {
		dst[i] = src[i]
		n--
	}
}

func ucopy(n int, dst, src []float64) {
	copy(dst[:n], src[:n])
}

func copy2d(shp, str nd.Shape, dst, src []float64) {
	n := shp[1]
	step0, step1 := str[0], str[1]
	if step1 > 1 {
		for i := 0; i < shp[0]; i++ {
			b := step0 * i
			icopy(n, dst[b:], src[b:], step1)
		}
		return
	}
	for i := 0; i < shp[0]; i++ {
		b := step0 * i
		ucopy(n, dst[b:], src[b:])
	}
}

func cpy(dst, src nd.Array) nd.Array {
	if !nd.IsShapeSame(dst, src) {
		panic("input arrays must have the same shape")
	}

	ndims := dst.Ndims()
	dshp, dstr := dst.Shape(), dst.Strides()
	dd := dst.Data()
	sd := src.Data()

	if ndims < 3 {
		copy2d(dshp, dstr, dd, sd)
		return dst
	}



	return dst
}
