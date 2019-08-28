package nd

func (array *ndarray) Data() []float64 { return array.data }

func (array *ndarray) Shape() Shape { return array.shape }

func (array *ndarray) Strides() Shape { return array.strides }

func (array *ndarray) Size() int { return array.size }

func (array *ndarray) Ndims() int { return array.ndims }

// TODO: Read copies the underlying data into dst
func (array *ndarray) Read(dst []byte) (n int, err error) { return }

// TODO: Write copies src into the underlying data
func (array *ndarray) Write(src []byte) (n int, err error) { return }

func (array *ndarray) At(n Index) *float64 { return &array.data[Sub2ind(array.strides, n)] }

func (array *ndarray) Get(n Index) float64 { return array.data[Sub2ind(array.strides, n)] }

func (array *ndarray) Set(v float64, n Index) { array.data[Sub2ind(array.strides, n)] = v }

// NewIter returns an nd-iterator for the array
func (array *ndarray) Iter() Iterator { return array.it }

func (array *ndarray) Take(ind Index, res ...Array) Array {
	var out Array
	if len(res) < 1 {
		out = Zeros(Shape{1, len(ind)})
	} else {
		if len(ind) > res[0].Size() {
			panic("Take: provided Array is not big enough")
		}
		out = res[0]
	}

	ad, ai := array.Range().Iter()
	od := out.Data()
	for k, l := range ind {
		od[k] = ad[ai[l]]
	}

	return out
}

func (array *ndarray) Range(rng ...int) Iterator {
	it := array.it

	switch len(rng) {
	case 0:
		it.beg = 0
		it.len = array.size
		it.inc = 1
	case 1:
		it.beg = rng[0]
		it.len = array.size - it.beg + 1
		it.inc = 1
	case 2:
		it.beg = rng[0]
		it.len = rng[1] - it.beg + 1
		it.inc = 1
	default:
		it.beg = rng[0]
		it.inc = rng[2]
		it.len = (rng[1] - it.beg) / it.inc
	}
	return it
}

func (array *ndarray) isView() bool {
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
