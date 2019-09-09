package nd

const (
	ErrShape     = "incompatible shapes"
	ErrBroadcast = "could not broadcast shapes to a single shape"
)

func (array *Ndarray) Data() []float64 { return array.data }

func (array *Ndarray) Shape() Shape { return array.shape }

func (array *Ndarray) Strides() Shape { return array.strides }

func (array *Ndarray) Size() int { return array.size }

func (array *Ndarray) Ndims() int { return array.ndims }

// TODO: Read copies the underlying data into dst
func (array *Ndarray) Read(dst []byte) (n int, err error) { return }

// TODO: Write copies src into the underlying data
func (array *Ndarray) Write(src []byte) (n int, err error) { return }

func (array *Ndarray) At(n Index) *float64 { return &array.data[Sub2ind(array.strides, n)] }

func (array *Ndarray) Get(n Index) float64 { return array.data[Sub2ind(array.strides, n)] }

func (array *Ndarray) Set(v float64, n Index) { array.data[Sub2ind(array.strides, n)] = v }

// Iter returns an nd-iterator for the array
func (array *Ndarray) Iter() Iterator { return array.it }

func (array *Ndarray) Take(ind Index, res ...Array) Array {
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

func (array *Ndarray) Range(rng ...int) Iterator {
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

func _icopy(n int, dst, src []float64, inc int) {
	for i := range src[:n] {
		i *= inc
		dst[i] = src[i]
	}
}

func _ucopy(n int, dst, src []float64) {
	for i, v := range src[:n] {
		dst[i] = v
	}
}

func _copy(shp, str Shape, dst, src []float64) {
	n := shp[1]
	inc := str[1]
	if inc > 1 {
		for i := 0; i < shp[0]; i++ {
			b := i * str[0]
			_icopy(n, dst[b:], src[b:], inc)
		}
		return
	}

	for i := 0; i < shp[0]; i++ {
		b := i * str[0]
		_ucopy(n, dst[b:], src[b:])
	}
}

func Copy(dst, src *Ndarray) *Ndarray {
	if !IsShapeSame(dst, src) {
		panic("dst and src must have same shape")
	}

	if dst.ndims < 3 {
		_copy(dst.shape, dst.strides, dst.data, src.data)
		return dst
	}

	nd := dst.ndims
	shp := dst.shape
	str := dst.strides

	ishp := make(Shape, nd)
	copy(ishp, shp)
	for k := nd - 2; k < nd; k++ {
		ishp[k] = 1
	}

	istr := ComputeStrides(ishp)
	iistr := computeInverseStrides(istr)
	for k := 0; k < ComputeSize(shp[:nd-1]); k++ {
		j := dst.ind(iistr, istr, k)
		_copy(shp[nd-2:], str[nd-2:], dst.data[j:], src.data[j:])
	}

	return dst
}

func (array *Ndarray) Clone() *Ndarray {
	res := Zeroslike(array)
	return res
}

func (array *Ndarray) ind(istr []float64, str Shape, k int) (s int) {
	for i, n := range istr {
		j := int(float64(k) * n)
		s += j * array.strides[i]
		k -= j * str[i]
	}
	return
}

func computeInverseStrides(str Shape) (istr []float64) {
	istr = make([]float64, 0, len(str))
	for _, n := range str {
		istr = append(istr, 1/float64(n))
	}
	return
}
