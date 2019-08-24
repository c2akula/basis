package nd

import (
	"strconv"
	"strings"
)

type Iterator interface {
	// Begin returns the start coordinate of the iterator.
	Begin() Index
	// End returns the stop coordinate of the iterator.
	// The end of the iterator is inclusive of the range [Begin, End].
	End() Index
	// Len returns the depth of the iterator. In other words,
	// the no. of iterations it will take to go from Begin to End.
	Len() int
	// Next advances the iterator by 1 step.
	Next()
	// At returns coordinate at (n+1)th step, such that At(0) == Begin() and At(Len()-1) == End().
	// Equivalent to Advance(n).I().
	At(n int) Index
	// Advance advances the iterator by n steps from current position.
	// It is equivalent to calling Next n times.
	Advance(n int) Iterator
	// Done checks if the iterator has reached the End.
	Done() bool
	// Reset resets the internal index to Begin.
	Reset()
	// I returns the location of the iterator as an n-d index.
	I() Index
	// Upk unpacks the value at the location of the iterator.
	Upk() *float64

	From(beg int) Iterator
	To(end int) Iterator
	WithStep(inc int) Iterator
}

type iterator struct {
	array         Array   // reference to the array being iterated upon
	strides       Shape   // iterator strides
	ind2submap    []Index // linear index to subscript map
	beg, end, ind int     // indices to keep track of the iteration
	inc           int     // increments
	len           int     // # of iterations to end
	ndims         int     // rank of the iterator
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
	it.ind = 0
	it.createInd2submap()
	return it
}

func (it *iterator) Next() {
	it.ind += it.inc
}

func (it *iterator) Done() bool {
	return it.ind > it.end
}

func (it *iterator) I() Index {
	return it.ind2submap[it.ind]
}

func (it *iterator) At(n int) Index {
	return it.ind2submap[n]
}

func (it *iterator) Advance(n int) Iterator {
	it.ind += n
	return it
}

func (it *iterator) Len() int { return it.len }

func (it *iterator) Begin() Index {
	return it.ind2submap[it.beg]
}

func (it *iterator) End() Index {
	return it.ind2submap[it.end]
}

// FIXME: Previously it was it.ind = it.beg
func (it *iterator) Reset() {
	// If ind is reset to beg, then functions
	// like Take will have to create a new iterator
	// every time they are called, since take overwrites
	// the array's iterator's parameters.
	// Resetting ind shouldn't be a problem, since
	// when an iterator is created, it is started at
	// 0, relative to the beginning of the array or view.
	// it.ind = it.beg
	it.beg = 0
	it.ind = 0
	// Similarly, previously there was no resetting the end
	// but, now since the take function overwrites the iterator's
	// parameters, we set it to the end of the array or view
	it.end = it.array.Size() - 1
}

func (it *iterator) Upk() *float64 {
	data := it.array.Data()
	return &data[sub2ind(it.array.Strides(), it.I())]
}

// Subs creates a non-array iterator between begin and end, inclusive.
// Note: It cannot be used to iterate over an array.
// FIXME: shape of the iterator should be computed
// relative to beg = 0.
func Subs(beg, end Index) Iterator {
	it := &iterator{len: 1, inc: 1}
	it.strides = make(Shape, len(end))
	for i := range it.strides {
		it.strides[i] = end[i] - beg[i] + 1 // shape
		it.len *= it.strides[i]
	}
	it.ndims = len(end)
	it.strides = ComputeStrides(it.strides)
	it.beg = 0
	it.end = sub2ind(it.strides, end)
	it.ind = sub2ind(it.strides, beg)
	it.createInd2submap()
	return it
}

func (it *iterator) From(beg int) Iterator {
	it.beg = beg
	it.ind = beg
	it.inc = 1
	return it
}

func (it *iterator) To(end int) Iterator {
	it.inc = 1
	it.end = end
	return it
}

func (it *iterator) WithStep(inc int) Iterator {
	it.inc = inc
	return it
}

func (it *iterator) createInd2submap() {
	it.ind2submap = make([]Index, it.len)
	subs := make(Index, it.len*it.ndims)
	for i := range it.ind2submap {
		it.ind2submap[i], subs = subs[:it.ndims], subs[it.ndims:]
		ind2sub(it.strides, i, it.ind2submap[i])
	}
}

func (it *iterator) String() string {
	var sb strings.Builder
	for _, v := range it.I() {
		sb.WriteString(strconv.Itoa(v))
	}
	return sb.String()
}
