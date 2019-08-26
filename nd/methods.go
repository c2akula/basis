package nd

func (array *ndarray) Data() []float64 { return array.data }

// Shape returns the dimensions of the go.nd
func (array *ndarray) Shape() Shape {
	return array.shape
}

func (array *ndarray) Strides() Shape { return array.strides }

// Size returns the no. of elements in the go.nd
func (array *ndarray) Size() int {
	return array.size
}

// Ndims returns the rank or dimensionality of the go.nd
func (array *ndarray) Ndims() int {
	return array.ndims
}

// TODO: Read copies the underlying data into dst
func (array *ndarray) Read(dst []byte) (n int, err error) { return }

// TODO: Write copies src into the underlying data
func (array *ndarray) Write(src []byte) (n int, err error) { return }

// Get returns the element at the coordinate n
func (array *ndarray) Get(n Index) float64 {
	return array.data[Sub2ind(array.strides, n)]
}

// Set sets the elment v at the coordinate n
func (array *ndarray) Set(v float64, n Index) {
	array.data[Sub2ind(array.strides, n)] = v
}

// Take returns an go.nd-iterator for the array
func (array *ndarray) Take() Iterator {
	return array.it
}

// TakeAt returns an flat array with the values at the indices
// specified by i.
func (array *ndarray) TakeAt(i Index) Array {
	arr := Zeros(Shape{1, len(i)})
	it := arr.Take()
	for _, v := range i {
		*it.At() = array.data[array.it.Seek(v)]
		it.Next()
	}
	it.Reset()
	return arr
}
