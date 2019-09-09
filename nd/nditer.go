package nd

// Nditer is an efficient multidimensional iterator for ndarray.
type Nditer struct {
	x      []float64 // array data
	xstr   Shape     // array strides
	istr   Shape     // iterator strides
	stri   []float64 // inverse iterator strides
	sub    Index     // track cartesian index to start of row
	b      int       // track linear index to start of row
	k      int       // track iterations
	sz     int       // # of iterations
	rs, rn int       // strides and sizes of dimensions returned
	depth  int       // controls external loop depth
}

// TODO: Provide a depth flag to control which
// dimension you'd like to iterate along and provide
// slices of. Example, Depth=0 would be the default
// behavior, returning rows, and iterations outside
// dimension n-1.

// NewNditer creates an nd-iterator for the given array.
func NewNditer(x *Ndarray) *Nditer {
	ndit := x.ndims - 1
	it := &Nditer{
		sub:  make(Index, ndit),
		sz:   ComputeSize(x.shape[:ndit]),
		istr: ComputeStrides(x.shape[:ndit]),
		stri: make([]float64, ndit),
		rn:   x.shape[ndit],
		rs:   x.strides[ndit],
		x:    x.data,
		xstr: x.strides[:ndit],
	}
	for i, n := range it.istr {
		it.stri[i] = 1 / float64(n)
	}
	return it
}

// Reset sets the iterator to the default state.
func (it *Nditer) Reset() { it.k = 0 }

// Sub returns the cartesian index of current row referenced
// by the iterator.
func (it *Nditer) Sub() Index { return it.sub }

// Ind returns the linear index to the start of the current
// row referenced by the iterator.
func (it *Nditer) Ind() int { return it.b }

// At returns a slice containing the kth row and its length.
func (it *Nditer) At(k int) ([]float64, int) {
	return it.x[it.ind(k):], it.rn
}

// Get returns a slice containing the row's elements at
// the iterator's current position and its length.
func (it *Nditer) Get() ([]float64, int) {
	it.k++
	return it.x[it.b:], it.rn
}

// Next checks if there are any rows left to iterate.
func (it *Nditer) Next() bool {
	it.b = it.ind(it.k)
	return it.k < it.sz
}

// Step advances the iterator by n iterations. If n is specified
// such that the iterator goes out of bounds, it will panic.
func (it *Nditer) Step(n int) *Nditer {
	if it.k+n > it.sz {
		panic("no. of steps specified is out of bounds")
	}
	it.k += n
	return it
}

// Stride returns the step size to use to access elements in the
// slice returned by Get.
func (it *Nditer) Stride() (n int) { return it.rs }

func (it *Nditer) ind(k int) (s int) {
	for i, n := range it.stri {
		j := int(float64(k) * n)
		s += j * it.xstr[i]
		k -= j * it.istr[i]
		it.sub[i] = j
	}
	return
}

// ZipIter provides an iterator to iterate over two Ndarrays
// simultaneously.
type ZipIter struct {
	x, y   *Ndarray  // arrays referenced by the iterator
	str    Shape     // iterator strides
	istr   []float64 // inverse iterator strides
	xb, yb int       // row tracking linear indices relative to arrays
	sub    Index     // row tracking cartesian index relative to iterator
	rn, rs int       // row length and stride
	k, sz  int       // track iterations
}

// Zip creates an iterator to iterate over the Ndarrays, x and y
// simultaneously.
func Zip(x, y *Ndarray) *ZipIter {
	// check if x and y are same shape
	xv := x.View(make(Index, x.ndims), x.shape)
	yv := y.View(make(Index, y.ndims), y.shape)
	if !IsShapeSame(x, y) {
		if !isBroadcastable(x, y) {
			panic("zip: cannot create zip iterator for the arrays. dimensions mismatch")
		}
		Broadcast(xv, yv)
	}

	it := &ZipIter{
		x:   x,
		y:   y,
		sz:  ComputeSize(x.shape[:x.ndims-1]),
		rn:  x.shape[x.ndims-1],
		rs:  x.strides[x.ndims-1],
		sub: make(Index, x.ndims),
	}

	ishp := make(Shape, it.x.ndims)
	copy(ishp, it.x.shape)
	ishp[it.x.ndims-1] = 1
	it.str = ComputeStrides(ishp)
	it.istr = make([]float64, it.x.ndims)
	for i, n := range it.str {
		it.istr[i] = 1 / float64(n)
	}
	return it
}

// Done checks if all the rows have been iterated
func (it *ZipIter) Done() bool { return it.k == it.sz }

// Reset sets the iterator to the default state.
func (it *ZipIter) Reset() { it.k = 0 }

// Sub returns the cartesian index to the start of the current row
// referenced by the iterator.
func (it *ZipIter) Sub() Index { return it.sub }

// Ind returns the linear index to the start of the current row
// referenced by the iterator.
func (it *ZipIter) Ind() (xi, yi int) { return it.xb, it.yb }

// Stride returns the step size to use to access elements
// from the slice returned by Next.
func (it *ZipIter) Stride() int { return it.rs }

// Next returns the rows referenced by the iterator at the current
// iteration and the length of the row.
func (it *ZipIter) Next() (xv, yv []float64, rn int) {
	it.xb, it.yb = it.ind(it.k)
	xv, yv = it.x.data[it.xb:], it.y.data[it.yb:]
	it.k++
	return xv, yv, it.rn
}

func (it *ZipIter) ind(k int) (xs, ys int) {
	for i, n := range it.istr[:it.x.ndims-1] {
		j := int(float64(k) * n)
		xs += j * it.x.strides[i]
		ys += j * it.y.strides[i]
		k -= j * it.str[i]
		it.sub[i] = j
	}
	return
}
