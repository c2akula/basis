package nd

const (
	ErrShape     = "incompatible shapes"
	ErrBroadcast = "could not broadcast shapes to a single shape"
)

// Data returns the array's internal buffer.
func (array *Ndarray) Data() []float64 { return array.data }

// Shape returns the sizes of each dimension.
func (array *Ndarray) Shape() Shape { return array.shape }

// Strides returns the strides along each dimension.
func (array *Ndarray) Strides() Shape { return array.strides }

// Size returns the no. of elements in array.
func (array *Ndarray) Size() int { return array.size }

// Ndims returns the dimensionality of array.
func (array *Ndarray) Ndims() int { return array.ndims }

// Read copies the underlying data into dst
func (array *Ndarray) Read(dst []byte) (n int, err error) { return }

// Write copies src into the underlying data.
func (array *Ndarray) Write(src []byte) (n int, err error) { return }

// At provides read-write access to the element at cartesian index n.
func (array *Ndarray) At(n Index) *float64 { return &array.data[Sub2ind(array.strides, n)] }

// Get returns the element at cartesian index n.
func (array *Ndarray) Get(n Index) float64 { return array.data[Sub2ind(array.strides, n)] }

// Set puts the value v at cartesian index n.
func (array *Ndarray) Set(v float64, n Index) { array.data[Sub2ind(array.strides, n)] = v }

// Iter creates a new iterator for the array.
func (array *Ndarray) Iter() *Iter { return NewIter(array) }

// Zip creates a zip iterator for array and y.
func (array *Ndarray) Zip(y *Ndarray) *ZipIter { return Zip(array, y) }

func (array *Ndarray) isView() bool {
	for k := range array.shape {
		str := 1
		for _, s := range array.shape[k+1:] {
			str *= s
		}
		if str != array.strides[k] {
			return true
		}
	}
	return false
}

func (array *Ndarray) Begin() Index { return array.beg }

func (array *Ndarray) End() Index { return array.end }
