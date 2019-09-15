package nd

// Iterator is any object that can provide external iteration for Ndarray.
type Iterator interface {
	// Next checks if there are further iterations left.
	Next() bool
	// Get advances the iterator and returns a slice of elements x
	// along with the no. of valid elements n and the stride, str, required
	// to access the elements.
	Get() (n int, x []float64, str int)
	// Reset sets the iterator to its initial state.
	Reset()
}

// Iter is the default Iterator for Ndarray. It provides efficient access to all the
// rows in an Ndarray.
type Iter struct {
	x            []float64 // array data referenced by Iter
	xstr         []int     // strides of the referenced array data
	rn, rs       int       // length and stride of the row returned by Get
	istr         []int     // iterator strides
	stri         []float64 // inverse iterator strides
	k, size      int       // track iterations
	b            int       // track linear index to the start of the row returned by Get
	sub          []int     // track cartesian index to the start of the row returned by Get
	iscontiguous bool
}

// NewIter constructs a new iterator for array, x.
func NewIter(x *Ndarray) *Iter {
	ndx := x.Ndims()
	xshp := x.Shape()
	xstr := x.Strides()
	it := &Iter{
		x:            x.Data(),
		xstr:         xstr,
		rn:           xshp[ndx-1],
		rs:           xstr[ndx-1],
		istr:         ComputeStrides(xshp[:ndx-1]),
		stri:         make([]float64, ndx-1),
		sub:          make([]int, ndx-1),
		size:         ComputeSize(xshp[:ndx-1]),
		iscontiguous: iscontiguous(x),
	}
	for i, n := range it.istr {
		it.stri[i] = 1 / float64(n)
	}
	return it
}

// Next checks if there are any more rows left.
func (it *Iter) Next() bool {
	it.b = it.Ind(it.k)
	return it.k < it.size
}

// Get advances the iterator and returns a row of an Ndarray on every iteration of the iterator.
// It also returns the no. of valid elements n in the row and the stride, str,
// required to access the elements.
func (it *Iter) Get() (n int, v []float64, str int) {
	it.k++
	return it.rn, it.x[it.b:], it.rs
}

// Reset resets the iterator to its initial state.
func (it *Iter) Reset() { it.k = 0 }

// At returns a slice v containing the elements of the row at the iteration k.
// It can be used to implement an external iterator over multiple Ndarrays of the same size.
// It also returns the no. of valid elements n in the row and the stride, str,
// required to access the elements.
func (it *Iter) At(k int) (n int, v []float64, str int) {
	if k > 0 {
		return it.rn, it.x[it.Ind(k):], it.rs
	}

	return it.rn, it.x, it.rs
}

// Adv advances the iterator by n steps.
func (it *Iter) Adv(n int) bool {
	d := it.k + n
	if d > it.size {
		return false
	}
	it.k = d
	it.b = it.Ind(it.k)
	return true
}

// Ind converts the iteration counter k into a linear index s to the start of the row,
// relative to the Ndarray referenced by the iterator.
func (it *Iter) Ind(k int) (s int) {
	if it.iscontiguous {
		return it.cind(k)
	}

	return it.pind(k)
}

// ZipInd computes the linear indices to the start of the rows in the iterators
// x and y. It can be used when iterating over two Ndarrays simultaneously.
func ZipInd(x, y *Iter, k int) (xs, ys int) {
	if x.size != y.size {
		panic("iterators must have same size")
	}

	if len(x.stri) != len(y.stri) {
		panic("dimensions mismatch. iterators not compatible")
	}

	switch {
	case x.iscontiguous && y.iscontiguous:
		return x.cind(k), y.cind(k)
	case x.iscontiguous && !y.iscontiguous:
		return x.cind(k), y.pind(k)
	case !x.iscontiguous && y.iscontiguous:
		return x.pind(k), y.cind(k)
	default:
		i := 0
		xj, yj := 0, 0
		xk, yk := k, k
		for nd := len(x.stri) - 1; nd != 0; nd-- {
			xj = int(float64(xk) * x.stri[i])
			yj = int(float64(yk) * y.stri[i])
			xs += xj * x.xstr[i]
			ys += yj * y.xstr[i]
			xk -= xj * x.istr[i]
			yk -= yj * y.istr[i]
			i++
		}
		xj = int(float64(xk) * x.stri[i])
		yj = int(float64(yk) * y.stri[i])
		xs += xj * x.xstr[i]
		ys += yj * y.xstr[i]
	}
	return
}

// contiguous ind
func (it *Iter) cind(k int) (s int) { return k * it.xstr[len(it.xstr)-2] }

// plain ind
func (it *Iter) pind(k int) (s int) {
	nd := len(it.stri)
	for i, n := range it.stri[:nd-1] {
		j := int(float64(k) * n)
		s += j * it.xstr[i]
		k -= j * it.istr[i]
	}
	i := nd - 1
	j := int(float64(k) * it.stri[i])
	s += j * it.xstr[i]
	return
}

func iscontiguous(x *Ndarray) bool {
	str := ComputeStrides(x.Shape())
	xstr := x.Strides()
	if len(str) != len(xstr) {
		return false
	}

	for i, n := range xstr {
		if n != str[i] {
			return false
		}
	}
	return true
}

// Ind2sub returns a cartesian index converted from linear index k.
func (it *Iter) Ind2sub(k int) []int {
	nd := len(it.stri)
	for i, n := range it.stri[:nd-1] {
		j := int(float64(k) * n)
		k -= j * it.istr[i]
		it.sub[i] = j
	}
	i := nd - 1
	it.sub[i] = int(float64(k) * it.stri[i])
	return it.sub
}

// Sub2ind returns a linear index converted from cartesian index sub.
func (it *Iter) Sub2ind(sub []int) (s int) {
	for i, n := range it.sub[:len(it.stri)] {
		s += it.xstr[i] * n
	}
	return
}

type ZipIter struct {
	x, y         []float64 // referenced array data
	xstr, ystr   []int     // array strides
	nd           int       // no. of dimensions of the arrays referenced
	xistr, yistr []int     // iterator strides
	xstri, ystri []float64 // inverse iterator strides
	k, size      int       // track iterations
	xv, yv       vec       // tmp struct to package rows from Get
	rn           int       // length of the rows returned by Get
	xrs, yrs     int       // stride of the rows returned by Get
	xb, yb       int       // linear index to the start of the rows
	xic, yic     bool      // is contiguous?
}

//
func Zip(x, y *Ndarray) *ZipIter {
	if !IsShapeSame(x, y) {
		if err := Broadcast(x, y); err != nil {
			panic(err)
		}
	}

	// shapes are same, so we'll compute the size of the iterator from
	// one of the arrays.
	// similarly, rn will be the same, and hence we'll pick from one of the arrays.
	// however, rs will be different in the case of broadcasting, hence we'll
	// return an intermediary structure wrapping the respective length, data and stride
	// of the iterator from Get.

	ndims := x.Ndims()
	xshp := x.Shape()
	yshp := y.Shape()
	xstr := x.Strides()
	ystr := y.Strides()

	it := &ZipIter{
		x:     x.Data(),
		y:     y.Data(),
		nd:    ndims,
		xstr:  xstr,
		ystr:  ystr,
		size:  ComputeSize(xshp[:ndims-1]),
		rn:    xshp[ndims-1],
		xrs:   xstr[ndims-1],
		yrs:   ystr[ndims-1],
		xistr: ComputeStrides(xshp[:ndims-1]),
		yistr: ComputeStrides(yshp[:ndims-1]),
		xic:   iscontiguous(x),
		yic:   iscontiguous(y),
	}
	it.xstri = make([]float64, ndims-1)
	it.ystri = make([]float64, ndims-1)
	for i, n := range it.xistr {
		it.xstri[i] = 1 / float64(n)
	}
	for i, n := range it.yistr {
		it.ystri[i] = 1 / float64(n)
	}

	it.xv = vec{data: it.x, size: xshp[ndims-1], step: xstr[ndims-1]}
	it.yv = vec{data: it.y, size: yshp[ndims-1], step: ystr[ndims-1]}
	return it
}

func (it *ZipIter) Next() bool {
	it.xb, it.yb = it.ind(it.k)
	return it.k < it.size
}

func (it *ZipIter) Get() (xv, yv vec) {
	it.k++
	it.xv.data = it.x[it.xb:]
	it.yv.data = it.y[it.yb:]
	return it.xv, it.yv
}

func (it *ZipIter) Reset() { it.k = 0 }

func (it *ZipIter) ind(k int) (xs, ys int) {
	switch {
	case it.xic && it.yic:
		return k * it.xstr[it.nd-2], k * it.ystr[it.nd-2]
	case it.xic && !it.yic:
		return k * it.xstr[it.nd-2], ind(it.ystr, it.yistr, it.ystri, k)
	case !it.xic && it.yic:
		return ind(it.xstr, it.xistr, it.xstri, k), k * it.ystr[it.nd-2]
	default:
		i := 0
		xj, yj := 0, 0
		xk, yk := k, k
		for nd := it.nd - 2; nd != 0; nd-- {
			xj = int(float64(xk) * it.xstri[i])
			yj = int(float64(yk) * it.ystri[i])
			xs += xj * it.xstr[i]
			ys += yj * it.ystr[i]
			xk -= xj * it.xistr[i]
			yk -= yj * it.yistr[i]
			i++
		}
		xj = int(float64(xk) * it.xstri[i])
		yj = int(float64(yk) * it.ystri[i])
		xs += xj * it.xstr[i]
		ys += yj * it.ystr[i]
	}
	return
}

type vec struct {
	data []float64
	size int
	step int
}

func ind(xstr, istr []int, stri []float64, k int) (s int) {
	nd := len(stri)
	for i, n := range stri[:nd-1] {
		j := int(float64(k) * n)
		s += j * xstr[i]
		k -= j * istr[i]
	}
	i := nd - 1
	j := int(float64(k) * stri[i])
	s += j * xstr[i]
	return
}
