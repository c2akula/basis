package nd

// Nditer is an efficient multidimensional iterator for ndarray.
type Nditer struct {
	str     Shape
	istr    []float64
	sub     Index
	k, b    int
	isplain bool
	*ndarray
}

// NewNditer creates an nd-iterator for the given array.
func NewNditer(array *ndarray) *Nditer {
	it := new(Nditer)
	it.ndarray = array
	it.sub = make(Index, it.ndims)
	it.isplain = !(it.strides[it.ndims-1] != 1)
	ishp := make(Shape, it.ndims)
	copy(ishp, it.shape)
	ishp[it.ndims-1] = 1
	it.str = ComputeStrides(ishp)
	it.istr = make([]float64, 0, it.ndims)
	for _, n := range it.str {
		it.istr = append(it.istr, 1/float64(n))
	}
	return it
}
