package nd

// Sub2ind converts an array subscript n into a linear index k.
func Sub2ind(array Array, n Index) (k int) {
	arr := array.(*ndarray)
	_ = n[arr.ndims-1]
	return sub2ind(arr.strides, n)
}

func sub2ind(strides Shape, n Index) (k int) {
	for i, s := range strides {
		k += s * n[i]
	}
	return
}

func ind2sub(strides Shape, k int, ind Index) Index {
	_ = ind[len(strides)-1]
	for j, s := range strides {
		l := int(float64(k) * (1.0 / float64(s)))
		k -= l * s
		ind[j] = l
	}
	return ind
}

func (res *ndarray) sub2ind(n Index) int { return sub2ind(res.strides, n) }
func (res *ndarray) ind2sub(k int) []int { return res.it.At(k) }

// ComputeStrides computes the offsets along each dimension from shape.
func ComputeStrides(shape Shape) Shape {
	strides := make(Shape, len(shape))
	for k := range shape {
		strides[k] = 1
		for _, n := range shape[k+1:] {
			strides[k] *= n
		}
	}
	return strides
}
