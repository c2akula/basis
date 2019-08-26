package nd

// Iterator is an object that provides a way to iterate over and access elements of an Array.
type Iterator interface {
	// Len returns the depth of the iterator.
	Len() int

	// Next advances the iterator by 1 step.
	Next()

	// Data provides access to the array's buffer that the iterator is referencing.
	Data() []float64

	// Seek returns the linear index referencing the internal array's buffer at the (n+1)th position of the iterator.
	Seek(n int) int

	// Advance advances the iterator by n steps from current position.
	// It is equivalent to calling Next n times.
	Advance(n int) Iterator

	// Done checks if the iterator has reached the End.
	Done() bool

	// Reset resets the internal index to Begin.
	Reset()

	// I returns the location of the iterator as a linear index referencing the internal array.
	I() int

	// At provides access to the value at the location of the iterator.
	At() *float64

	// Indices returns the linear indices that reference into the internal array's elements.
	Indices() Index

	// Cur returns the iterator's position as an n-dimensional index.
	Cur() Index

	From(beg int) Iterator
	To(end int) Iterator
	WithStep(inc int) Iterator
}

type iterator struct {
	array         Array // reference to the array being iterated upon
	strides       Shape // iterator strides
	beg, end, ind int   // indices to keep track of the iteration
	inc           int   // increment
	len           int   // # of iterations to end
	ndims         int   // rank of the iterator
	ind2submap    Index // linear index to subscript map
	sub2indmap    Index // subscript to linear index map
}

func Iter(array Array) Iterator {
	it := &iterator{
		array: array,
		len:   array.Size(),
		inc:   1,
	}
	it.strides = ComputeStrides(it.array.Shape())
	it.ndims = array.Ndims()
	it.beg = 0
	it.end = array.Size() - 1
	it.ind = it.beg
	it.createInd2submap()
	it.createsub2indmap()
	return it
}

func (it *iterator) Next() {
	it.ind += it.inc
}

func (it *iterator) Done() bool {
	return it.ind > it.end
}

func (it *iterator) I() int { return it.sub2indmap[it.ind] }

func (it *iterator) Seek(n int) int { return it.sub2indmap[n] }

func (it *iterator) Data() []float64 { return it.array.Data() }

func (it *iterator) Advance(n int) Iterator {
	it.ind += n
	return it
}

func (it *iterator) Len() int { return it.len }

// Reset resets the state of the iterator.
func (it *iterator) Reset() {
	it.beg = 0
	it.ind = 0
	it.end = it.array.Size() - 1
	it.inc = 1
	it.len = it.array.Size()
}

func (it *iterator) At() *float64 {
	array := it.array.(*ndarray)
	return &array.data[it.sub2indmap[it.ind]]
}

func (it *iterator) Indices() Index { return it.sub2indmap }

func (it *iterator) Cur() Index {
	b := it.ind * it.ndims
	return it.ind2submap[b : b+it.ndims]
}

func (it *iterator) From(beg int) Iterator {
	it.beg = beg
	it.ind = beg
	it.inc = 1
	it.len = it.end - it.beg + 1
	return it
}

func (it *iterator) To(end int) Iterator {
	it.inc = 1
	it.end = end
	it.len = it.end - it.beg + 1
	return it
}

func (it *iterator) WithStep(inc int) Iterator {
	it.inc = inc
	q := (it.end - it.beg + 1) / it.inc
	r := (it.end - it.beg + 1) % it.inc
	it.len = q + r
	return it
}

func (it *iterator) createInd2submap() {
	// it.ind2submap = make([]Index, it.len)
	it.ind2submap = make(Index, it.len*it.ndims)
	// subs := make(Index, it.len*it.ndims)
	for i := range it.ind2submap[:it.len] {
		// it.ind2submap[i], subs = subs[:it.ndims], subs[it.ndims:]
		// Ind2sub(it.strides, i, it.ind2submap[i])
		b := it.ndims * i
		Ind2sub(it.strides, i, it.ind2submap[b:])
	}
}

func (it *iterator) createsub2indmap() {
	it.sub2indmap = make(Index, it.len)
	ind := make(Index, it.ndims)
	for i := 0; i < it.len; i++ {
		it.sub2indmap[i] = Sub2ind(it.array.Strides(), Ind2sub(it.strides, i, ind))
	}
}
