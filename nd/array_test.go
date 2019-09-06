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
