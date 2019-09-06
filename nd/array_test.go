package nd

import (
	"fmt"
	"testing"
)

// var TestArrayShape = Shape{3, 45, 15}

// var TestArrayShape = Shape{10, 45, 30} // 13,500
// var TestArrayShape = Shape{300, 45} // 13,500
var TestArrayShape = Shape{100, 100, 100}

// var TestArrayShape = Shape{1e3, 1e3}

// var TestArrayShape = Shape{10, 10, 10}

func reshape(array *ndarray, shp Shape) *ndarray {
	if ComputeSize(shp) != array.size {
		panic("new shape should compute to the same size as the original")
	}

	if array.isView() {
		res := Zeros(shp).(*ndarray)
		return Copy(res, array)
	}

	// array.View(make(Index, array.ndims), array.shape).(*ndarray)
	array.ndims = len(shp)
	array.shape = array.shape[:0]
	array.shape = array.shape[:array.ndims]
	copy(array.shape, shp)
	// array.strides = array.strides[:0]
	// array.strides = array.strides[:array.ndims]
	// computestrides(shp, array.strides[:array.ndims])
	return array
}

const ErrBroadcast = "broadcasting failed, dimensions are not compatible"

func isbroadcastable(n int, a, b Shape) bool {
	nda := len(a)
	ndb := len(b)
	for k, j := ndb-1, nda-1; n != 0; k, j = k-1, j-1 {
		nb, na := b[k], a[j]
		if nb != na {
			if !(nb == 1 || na == 1) {
				return false
			}
		}
		n--
	}
	return true
}

func broadcastShape(a, b Shape) (Shape, error) {
	// find the smaller shape
	ndr := 0
	nda, ndb := len(a), len(b)
	sza, szb := ComputeSize(a), ComputeSize(b)
	n := 0 // ndims of the smaller shape
	// set the size of the new shape to the larger of the two input shapes
	if sza > szb {
		ndr = nda
		n = ndb
	} else {
		ndr = ndb
		n = nda
	}

	// check if dimensions are compatible
	// 1. dimensions are equal, or
	// 2. one of the dimensions is 1
	if !isbroadcastable(n, a, b) {
		fmt.Printf("got=(a=%v,b=%v)\n", a, b)
		return nil, fmt.Errorf("%v", ErrBroadcast)
	}

	// make storage for new shape
	r := make(Shape, ndr)
	// copy the smaller shape into r in reverse order
	if sza > szb {
		// copy b
		for k, j := ndb-1, ndr-1; k >= 0; k, j = k-1, j-1 {
			r[j] = b[k]
		}
		// compare and set dimensions
		for k, na := range a {
			if nr := r[k]; na > nr {
				r[k] = na
			} else if r[k] == 0 {
				r[k] = 1
			}
		}

	} else {
		// copy a
		for k, j := nda-1, ndr-1; k >= 0; k, j = k-1, j-1 {
			r[j] = a[k]
		}
		// compare and set dimensions
		for k, nb := range b {
			if nr := r[k]; nb > nr {
				r[k] = nb
			} else if r[k] == 0 {
				r[k] = 1
			}
		}
	}

	return r, nil
}

func broadcastStrides(shp, bshp Shape) Shape {
	bstr := ComputeStrides(bshp)
	str := ComputeStrides(shp)
	if nshp, nbshp := len(shp), len(bshp); nshp < nbshp {
		for k, j := nshp-1, nbshp-1; k >= 0; k, j = k-1, j-1 {
			bstr[j] = str[k]

			if shp[k] == 1 {
				bstr[j] = 0
			}
		}

		// set the strides at the dimensions outside shp to 0
		for i := 0; i < nbshp-nshp; i++ {
			bstr[i] = 0
		}
		return bstr
	}

	copy(bstr, str)
	for j, n := range shp {
		if n == 1 {
			bstr[j] = 0
		}
	}
	return bstr
}

func TestBroadcastStrides(t *testing.T) {
	shp := Shape{2, 4, 3, 5}
	x := Reshape(Arange(0, float64(ComputeSize(shp))), shp).(*ndarray)
	fmt.Println("x: ", x)
	shp = Shape{1, 3, 5}
	y := Reshape(Arange(0, float64(ComputeSize(shp))), shp).(*ndarray)
	fmt.Println("y: ", y)
	bs, err := broadcastShape(x.shape, y.shape)
	if err != nil {
		panic(err)
	}
	bstr := broadcastStrides(y.shape, bs)
	fmt.Println("bstr: ", bstr)
}

func TestBroadcasting(t *testing.T) {
	shp := Shape{4, 3}
	x := Reshape(Arange(0, float64(ComputeSize(shp))), shp).(*ndarray)
	fmt.Println("x: ", x)
	y := Arange(0, 3).(*ndarray)
	fmt.Println("y: ", y)

	// TODO: broadcast strides
	shapes := []struct {
		xshp, yshp, rshp Shape
		err              error
	}{
		{Shape{256, 256, 3}, Shape{3}, Shape{256, 256, 3}, nil},
		{Shape{8, 1, 6, 1}, Shape{7, 1, 5}, Shape{8, 7, 6, 5}, nil},
		{Shape{7, 1, 5}, Shape{8, 1, 6, 1}, Shape{8, 7, 6, 5}, nil},
		{Shape{5, 4}, Shape{1}, Shape{5, 4}, nil},
		{Shape{1}, Shape{5, 4}, Shape{5, 4}, nil},
		{Shape{5, 4}, Shape{5, 4}, Shape{5, 4}, nil},
		{Shape{15, 3, 5}, Shape{15, 1, 5}, Shape{15, 3, 5}, nil},
		{Shape{4, 3, 5}, Shape{4, 1, 3}, nil, fmt.Errorf("%v", ErrBroadcast)},
		{Shape{15, 3, 5}, Shape{15, 3, 5}, Shape{15, 3, 5}, nil},
		{Shape{15, 3, 5}, Shape{3, 1}, Shape{15, 3, 5}, nil},
		{Shape{3, 1}, Shape{15, 3, 5}, Shape{15, 3, 5}, nil},
		{Shape{3}, Shape{4}, nil, fmt.Errorf("%v", ErrBroadcast)},
		{Shape{4}, Shape{3}, nil, fmt.Errorf("%v", ErrBroadcast)},
		{Shape{2, 1}, Shape{8, 4, 3}, nil, fmt.Errorf("%v", ErrBroadcast)},
		{Shape{8, 4, 3}, Shape{2, 1}, nil, fmt.Errorf("%v", ErrBroadcast)},
	}

	for i, shp := range shapes {
		bs, err := broadcastShape(shp.xshp, shp.yshp)
		if err != nil && shp.err != nil {
			if err.Error() != shp.err.Error() {
				t.Logf("test 'broadcast'=(%d) failed. got: %v, exp: %v\n", i, bs, shp.rshp)
				t.Fail()
			}
		}
		fmt.Printf("i:%d = bs: %v\n", i, bs)
	}

	A := Reshape(Arange(0, 6), Shape{2, 3}).(*ndarray)
	B := New(Shape{2, 1}, []float64{1, 2}).(*ndarray)
	fmt.Println("A: ", A)
	bs, err := broadcastShape(A.shape, B.shape)
	if err != nil {
		panic(err)
	}
	B.shape = bs
	B.strides = Shape{1, 0}
	fmt.Println("B: ", B)
	bit := newnditer(B)

	for inc := bit.Inc(); !bit.Done(); {
		v, n := bit.Next()
		for k := 0; k < n; k++ {
			j := k * inc
			fmt.Println("v: ", v[j], "inc: ", inc)
		}
	}

	fmt.Println(ComputeStrides(Shape{4, 3, 5}))
}

func TestNdarrayReshape(t *testing.T) {
	shp := Shape{2, 3, 4, 5}
	x := Reshape(Arange(0, float64(ComputeSize(shp))), shp).(*ndarray)
	fmt.Println("x: ", x)
	reshape(x, Shape{24, 5})
	fmt.Println("x*: ", x)
}

func TestNdarray_View(t *testing.T) {
	a := New(Shape{2, 2, 2, 3}, []float64{
		// t = 0, p = 0
		1, 2, 3, // r = 0, c = 0:2
		4, 5, 6, // r = 1, c = 0:2
		// t = 0, p = 1
		7, 8, 9, // r = 0, c = 0:2
		2, 0, 1, // r = 1, c = 0:2

		// t = 1, p = 0
		6, 4, 5,
		3, 1, 2,
		// t = 1, p = 1
		9, 7, 8,
		1, 0, 2,
	})
	b := a.View(
		Index{1, 0, 1, 0},
		Shape{2, 1, 3},
	)
	fmt.Println("b: ", b)

	exp := []float64{3, 1, 2, 1, 0, 2}
	elm := make([]float64, 0, len(exp))
	bd, bi := b.Range().Iter()
	for _, k := range bi {
		elm = append(elm, bd[k])
	}

	for i, v := range exp {
		if elm[i] != v {
			t.Logf("test failed. exp: %v, got: %v\n", exp, elm)
			t.Fail()
		}
	}
}

func TestNdarray_Get(t *testing.T) {
	a := New(Shape{2, 2, 3}, []float64{
		// p = 0
		1, 2, 3, // r = 0, c = 0:2
		4, 5, 6, // r = 1, c = 0:2
		// p = 1
		7, 8, 9, // r = 0, c = 0:2
		2, 0, 1, // r = 1, c = 0:2
	})

	b := a.View(Index{0, 1, 0}, Shape{2, 1, 3})
	exp := []float64{4, 5, 6, 2, 0, 1}
	elm := make([]float64, len(exp))
	ci := []Index{
		{0, 0, 0},
		{0, 0, 1},
		{0, 0, 2},
		{1, 0, 0},
		{1, 0, 1},
		{1, 0, 2},
	}
	for i := range exp {
		elm[i] = b.Get(ci[i])
	}

	for i, v := range exp {
		if elm[i] != v {
			t.Logf("test failed. exp: %v, got: %v\n", exp, elm)
			t.Fail()
		}
	}
}

func TestSub2ind(t *testing.T) {
	shape := []int{3, 4, 5}
	strides := ComputeStrides(shape)
	sub := [][]int{
		{0, 1, 2},
		{0, 1, 3},
		{0, 1, 4},
		{0, 2, 2},
		{0, 2, 3},
		{0, 2, 4},
		{0, 3, 2},
		{0, 3, 3},
		{0, 3, 4},

		{1, 1, 2},
		{1, 1, 3},
		{1, 1, 4},
		{1, 2, 2},
		{1, 2, 3},
		{1, 2, 4},
		{1, 3, 2},
		{1, 3, 3},
		{1, 3, 4},

		{2, 1, 2},
		{2, 1, 3},
		{2, 1, 4},
		{2, 2, 2},
		{2, 2, 3},
		{2, 2, 4},
		{2, 3, 2},
		{2, 3, 3},
		{2, 3, 4},
	}

	exp := []int{7, 8, 9, 12, 13, 14, 17, 18, 19, 27, 28, 29, 32, 33, 34, 37, 38, 39, 47, 48, 49, 52, 53, 54, 57, 58, 59}

	a := &ndarray{ndims: 3, strides: strides}

	for i, ind := range sub {
		if v := Sub2ind(a.strides, ind); exp[i] != v {
			t.Logf("test failed. exp: %v, got: %v\n", exp[i], v)
		}
	}
}

func TestNdarray_Iterator(t *testing.T) {
	a := New(Shape{2, 2, 2, 3}, []float64{
		// t = 0, p = 0
		1, 2, 3, // r = 0, c = 0:2
		4, 5, 6, // r = 1, c = 0:2
		// t = 0, p = 1
		7, 8, 9, // r = 0, c = 0:2
		2, 0, 1, // r = 1, c = 0:2

		// t = 1, p = 0
		6, 4, 5,
		3, 1, 2,
		// t = 1, p = 1
		9, 7, 8,
		1, 0, 2,
	})
	b := a.View(
		Index{1, 0, 1, 0},
		Shape{1, 2, 1, 3},
	)

	exp := []float64{3, 1, 2, 1, 0, 2}
	elm := make([]float64, 0, len(exp))
	bd, bi := b.Range().Iter()
	for _, k := range bi {
		elm = append(elm, bd[k])
	}

	for i, v := range exp {
		if elm[i] != v {
			t.Logf("test failed. exp: %v, got: %v\n", exp, elm)
			t.Fail()
		}
	}
}

func TestArange(t *testing.T) {
	got := Arange(0, 11)
	exp := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	it := got.Iter()
	gd := it.Data()
	for i, k := range it.Ind() {
		if exp[i] != gd[k] {
			t.Logf("test failed. exp: %v, got: %v\n", exp, got)
			t.Fail()
		}
	}
}

func TestReshape(t *testing.T) {
	got := Reshape(New(Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6}), Shape{1, 6})
	exp := New(Shape{1, 6}, []float64{1, 2, 3, 4, 5, 6})
	if got.String() != exp.String() {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func TestNdarray_Take(t *testing.T) {
	a := Reshape(Arange(0, 24), Shape{2, 4, 3})
	exp := []float64{1, 4, 3, 7}
	b := a.Take(Index{1, 4, 3, 7})
	bd, bi := b.Range().Iter()
	for i, k := range bi {
		if bd[k] != exp[i] {
			t.Logf("test 'Take' failed. exp: %v\n, got: %v\n", exp, bd)
			t.Fail()
		}
	}

	av := a.View(Index{0, 1, 1}, Shape{3, 2})
	b = av.Take(Index{2, 4, 3})
	exp = []float64{7, 10, 8}
	bd, bi = b.Range().Iter()
	for i, k := range bi {
		if bd[k] != exp[i] {
			t.Logf("test 'Take' failed. exp: %v\n, got: %v\n", exp, bd)
			t.Fail()
		}
	}
}

// Benchmarks

func BenchmarkInd2sub(b *testing.B) {
	b.ReportAllocs()
	a := Rand(TestArrayShape)
	strides := a.Strides()
	ind := make(Index, a.Ndims())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Ind2sub(strides, 1999, ind)
	}
	_ = ind[0]
}

func BenchmarkNdarray_Iterator(b *testing.B) {
	b.ReportAllocs()
	a := Rand(TestArrayShape)
	m := a.View(make(Index, a.Ndims()), TestArrayShape)
	md, mi := m.Range().Iter()
	s := 0.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, k := range mi {
			s = md[k]
		}
	}
	_ = s * s
}

func BenchmarkNdarray_View(b *testing.B) {
	b.ReportAllocs()
	a := Rand(TestArrayShape)
	beg := make(Index, a.Ndims())
	shape := TestArrayShape
	var m Array
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m = a.View(beg, shape)
	}
	_ = m
}

func BenchmarkNdarray_Get(b *testing.B) {
	a := Rand(TestArrayShape)
	ind := Index{2, 14, 3}
	v := 0.0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v = a.Get(ind)
	}
	_ = v * v
}
