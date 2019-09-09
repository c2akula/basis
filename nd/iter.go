package nd

// Iterator provides the ability to access the elements of and iterate over an Array.
type Iterator interface {
	// Len returns the size of the iterator, or the no. of elements referenced by the iterator.
	Len() int

	// Ind returns the positions of the elements referenced by the iterator.
	Ind() Index

	// Data returns the data buffer referenced by the iterator.
	Data() []float64

	Iter() ([]float64, Index)

	// Shape() Shape
	// Strides() Shape
	// Ndims() int
}

type iter struct {
	len int      // depth of the iterator
	ind Index    // list of indices into the referenced data buffer
	arr *Ndarray // reference to the array's data buffer iterating upon
	beg int      // range start
	inc int      // range step
	str Shape    // view strides computed from its actual shape
	sub Index    // scratch space for subscript computation using ind2sub
}

func (it *iter) Data() []float64 { return it.arr.data }

func (it *iter) Ind() Index {
	if it.ind != nil {
		return it.ind
	}

	it.ind = make(Index, 0, it.arr.size)
	it.sub = make(Index, it.arr.ndims)
	it.str = ComputeStrides(it.arr.shape)

	if it.arr.isView() {
		it.computeIndices(true)
	} else {
		it.computeIndices(false)
	}

	return it.ind
}

func (it *iter) Len() int { return it.len }

// func (it *iter) Shape() Shape             { return it.arr.Shape() }
// func (it *iter) Strides() Shape           { return it.arr.Strides() }
// func (it *iter) Ndims() int               { return it.arr.Ndims() }
func (it *iter) Iter() ([]float64, Index) {
	return it.arr.data, it.Ind()
}

// NewIter creates an iterator for the specified array.
func newiter(array *Ndarray) *iter {
	return &iter{
		arr: array,
		len: array.size,
		beg: 0,
		inc: 1,
	}
}

func (it *iter) computeIndices(forview bool) {
	if forview {
		for i := it.beg; i < it.len; i += it.inc {
			it.ind = append(it.ind, it.arr.sub2ind(Ind2sub(it.str, i, it.sub)))
		}
		return
	}
	for i := it.beg; i < it.len; i += it.inc {
		it.ind = append(it.ind, i)
	}
}
