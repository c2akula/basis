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

func testIsSliceIntSame(a, b []int) bool {
	for i, v := range a {
		if b[i] != v {
			return false
		}
	}
	return true
}

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
	strides := []struct {
		shp, bshp, bstr Shape
	}{
		{Shape{256, 256, 3}, Shape{256, 256, 3}, Shape{768, 3, 1}},
		{Shape{7, 1, 5}, Shape{8, 7, 6, 5}, Shape{0, 5, 0, 1}},
		{Shape{1}, Shape{5, 4}, Shape{0, 0}},
		{Shape{4, 1}, Shape{4, 8}, Shape{1, 0}},
		{Shape{5, 4}, Shape{5, 4}, Shape{4, 1}},
		{Shape{15, 1, 5}, Shape{15, 3, 5}, Shape{5, 0, 1}},
		{Shape{3, 1}, Shape{15, 3, 5}, Shape{0, 1, 0}},
	}

	for i, str := range strides {
		bstr := broadcaststr(str.shp, ComputeStrides(str.shp), str.bshp)
		if !testIsSliceIntSame(bstr, str.bstr) {
			t.Logf("test 'BroadcastStrides'=(%d) failed. got: %v, exp: %v\n", i, bstr, str.bstr)
			t.Fail()
		}
	}
}

func TestBroadcast(t *testing.T) {
	shapes := []struct {
		nmin             int
		isb              bool
		xshp, yshp, rshp Shape
	}{
		{1, true, Shape{256, 256, 3}, Shape{3}, Shape{256, 256, 3}},
		{3, true, Shape{8, 1, 6, 1}, Shape{7, 1, 5}, Shape{8, 7, 6, 5}},
		{3, true, Shape{7, 1, 5}, Shape{8, 1, 6, 1}, Shape{8, 7, 6, 5}},
		{1, true, Shape{5, 4}, Shape{1}, Shape{5, 4}},
		{1, true, Shape{1}, Shape{5, 4}, Shape{5, 4}},
		{1, true, Shape{8}, Shape{4, 1}, Shape{4, 8}},
		{2, true, Shape{5, 4}, Shape{5, 4}, Shape{5, 4}},
		{3, true, Shape{15, 3, 5}, Shape{15, 1, 5}, Shape{15, 3, 5}},
		{3, false, Shape{4, 3, 5}, Shape{4, 1, 3}, nil},
		{3, true, Shape{15, 3, 5}, Shape{15, 3, 5}, Shape{15, 3, 5}},
		{2, true, Shape{15, 3, 5}, Shape{3, 1}, Shape{15, 3, 5}},
		{2, true, Shape{3, 1}, Shape{15, 3, 5}, Shape{15, 3, 5}},
		{1, false, Shape{3}, Shape{4}, nil},
		{1, false, Shape{4}, Shape{3}, nil},
		{2, false, Shape{2, 1}, Shape{8, 4, 3}, nil},
		{2, false, Shape{8, 4, 3}, Shape{2, 1}, nil},
	}

	for i, shp := range shapes {
		bs := make(Shape, len(shp.rshp))
		szx, szy := ComputeSize(shp.xshp), ComputeSize(shp.yshp)
		if isb := isbroadcastable(shp.nmin, shp.xshp, shp.yshp); (isb == shp.isb) && shp.isb {
			broadcastshp(shp.xshp, szx, shp.yshp, szy, bs)
			if !testIsSliceIntSame(bs, shp.rshp) {
				t.Logf("test 'broadcast'=(%d) failed. got: %v, exp: %v\n", i, bs, shp.rshp)
				t.Fail()
			}
			fmt.Printf("i:%d = bs: %v\n", i, bs)
		}
	}
}

func TestNdarrayReshape(t *testing.T) {
	shp := Shape{2, 3, 4, 5}
	x := Reshape(Arange(0, float64(ComputeSize(shp))), shp)
	fmt.Println("x: ", x)
	x.Reshape(Shape{24, 5})
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

	for it := b.Iter(); it.Next(); {
		n, bd, str := it.Get()
		for j := 0; n != 0; j += str {
			elm = append(elm, bd[j])
			n--
		}
	}

	for i, v := range exp {
		if elm[i] != v {
			t.Logf("test failed. exp: %v, got: %v\n", exp, elm)
			t.Fail()
		}
	}
}

func TestInd2sub(t *testing.T) {
	shp := Shape{3, 4, 5}
	x := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
	ind := make(Index, x.ndims)
	stri := make([]float64, x.ndims)
	for j, n := range x.strides {
		stri[j] = 1 / float64(n)
	}
	for k := 0; k < x.size; k++ {
		fmt.Println("ind: ", Ind2sub(x.strides, k, ind), "sub: ", ind2ind(x.strides, stri, k))
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

	a := &Ndarray{ndims: 3, strides: strides}

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
	for bit := b.Iter(); bit.Next(); {
		n, bd, str := bit.Get()
		for j := 0; n != 0; j += str {
			elm = append(elm, bd[j])
			n--
		}
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
	for it := got.Iter(); it.Next(); {
		n, gd, _ := it.Get()
		for j, gv := range gd[:n] {
			if gv != exp[j] {
				t.Logf("test failed. exp: %v, got: %v\n", exp, got)
				t.Fail()
			}
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
	a := Rand(Shape{10, 10, 10, 1e2, 10})
	ind := Index{5, 5, 5, 34, 5}
	k := Sub2ind(a.strides, ind)
	stri := make([]float64, a.ndims)
	for j, n := range a.strides {
		stri[j] = 1 / float64(n)
	}
	v := 0.0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// v = a.Get(ind)
		// _ = Sub2ind(a.strides, ind)
		_ = ind2ind(a.strides, stri, k)
	}
	_ = v * v
}
