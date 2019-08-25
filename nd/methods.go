package nd

func (res *ndarray) Data() []float64 { return res.data }

// Shape returns the dimensions of the go.nd
func (res *ndarray) Shape() Shape {
	return res.shape
}

func (res *ndarray) Strides() Shape { return res.strides }

// Size returns the no. of elements in the go.nd
func (res *ndarray) Size() int {
	return res.size
}

// Ndims returns the rank or dimensionality of the go.nd
func (res *ndarray) Ndims() int {
	return res.ndims
}

// TODO: Read copies the underlying data into dst
func (res *ndarray) Read(dst []byte) (n int, err error) { return }

// TODO: Write copies src into the underlying data
func (res *ndarray) Write(src []byte) (n int, err error) { return }

// Get returns the element at the coordinate n
func (res *ndarray) Get(n Index) float64 {
	return res.data[sub2ind(res.strides, n)]
}

// Set sets the elment v at the coordinate n
func (res *ndarray) Set(v float64, n Index) {
	res.data[sub2ind(res.strides, n)] = v
}

// Take returns an go.nd-iterator for the array
func (res *ndarray) Take() Iterator {
	return res.it
}

// TakeAt returns an flat array with the values at the indices
// specified by i.
func (res *ndarray) TakeAt(i Index) Array {
	arr := Zeros(Shape{1, len(i)})
	it := arr.Take()
	for _, v := range i {
		*it.Upk() = res.Get(res.it.At(v))
		it.Next()
	}
	it.Reset()
	return arr
}
