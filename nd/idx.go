package nd

// Sub2ind converts a cartesian coordinate, ind into a linear index k.
func Sub2ind(strides Shape, ind Index) (k int) {
	for i, s := range strides {
		k += s * ind[i]
	}
	return
}

// Ind2sub converts a linear index k into cartesian coordinate, ind.
func Ind2sub(strides Shape, k int, ind Index) Index {
	for j, s := range strides {
		l := int(float64(k) * (1.0 / float64(s)))
		k -= s * l
		ind[j] = l
	}
	return ind
}

func (array *ndarray) sub2ind(n Index) int            { return Sub2ind(array.strides, n) }
func (array *ndarray) ind2sub(k int, sub Index) Index { return Ind2sub(array.strides, k, sub) }

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
