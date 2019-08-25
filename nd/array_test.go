package nd

import (
	"math/rand"
	"testing"
)

// var TestArrayShape = Shape{3, 45, 15}
// var TestArrayShape = Shape{10, 45, 30} // 13,500
var TestArrayShape = Shape{300, 45} // 13,500

func TestNdarray_String(t *testing.T) {
	// a := New(Shape{2, 2, 2, 3}, []float64{
	// 	// t = 0, p = 0
	// 	1, 2, 3, // r = 0, c = 0:2
	// 	4, 5, 6, // r = 1, c = 0:2
	// 	// t = 0, p = 1
	// 	7, 8, 9, // r = 0, c = 0:2
	// 	2, 0, 1, // r = 1, c = 0:2
	//
	// 	// t = 1, p = 0
	// 	6, 4, 5,
	// 	3, 1, 2,
	// 	// t = 1, p = 1
	// 	9, 7, 8,
	// 	1, 0, 2,
	// })
	a := Reshape(Arange(0, 60), Shape{3, 4, 5})
	// _ = a.String()
	exp := []float64{5, 6, 7, 25, 26, 27}
	b := a.View(
		Index{0, 1, 0},
		Shape{2, 1, 3},
	)
	it := Iter(b)
	for _, v := range exp {
		if b.Get(it.I()) != v {
			t.Logf("test failed. exp: %v\n, got: %v\n", exp, b)
		}
		it.Next()
	}
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

	exp := []float64{3, 1, 2, 1, 0, 2}
	elm := make([]float64, 0, len(exp))
	for it := Iter(b); !it.Done(); it.Next() {
		elm = append(elm, b.Get(it.I()))
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
	it := Iter(b)
	for i := range exp {
		elm[i] = b.Get(it.I())
		it.Next()
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
		if v := Sub2ind(a, ind); exp[i] != v {
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
	it := Iter(b)
	for ; !it.Done(); it.Next() {
		elm = append(elm, b.Get(it.I()))
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
	// fmt.Println(got)
	it := got.Take()
	for _, v := range exp {
		if *it.Upk() != v {
			t.Logf("test failed. exp: %v, got: %v\n", exp, got)
			t.Fail()
		}
		it.Next()
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

func BenchmarkInd2sub2(b *testing.B) {
	b.ReportAllocs()
	a := Rand(TestArrayShape)
	ind := make(Index, a.Ndims())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ind2sub(a.Strides(), 1999, ind)
	}
	_ = ind[0]
}

func BenchmarkNdarray_Iterator(b *testing.B) {
	b.ReportAllocs()
	a := Rand(Shape{5, 36, 16})
	m := a.View(Index{0, 0, 0}, TestArrayShape)
	it := Iter(m)
	s := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for it.Reset(); !it.Done(); it.Next() {
			s = it.Len()
		}
	}
	_ = s * s
}

func BenchmarkNdarray_View(b *testing.B) {
	b.ReportAllocs()
	a := Rand(TestArrayShape)
	beg := Index{0, 0, 0}
	shape := Shape{2, 15, 7}
	var m Array
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m = a.View(beg, shape)
	}
	_ = m
}

func BenchmarkNdarray_Get(b *testing.B) {
	a := Rand(TestArrayShape)
	it := a.Take()
	// ind := Index{2, 14, 3}
	v := 0.0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v = a.Get(it.I())
	}
	_ = v * v
}

func TestAdd(t *testing.T) {
	a := Reshape(Arange(0, 15), Shape{3, 5})
	b := Reshape(Arange(0, 15), Shape{3, 5})
	exp := Reshape(Arange(0, 15), Shape{3, 5})

	add := func(x, y Iterator) {
		for !y.Done() {
			*y.Upk() += *x.Upk()
			x.Next()
			y.Next()
		}
		x.Reset()
		y.Reset()
	}
	add(a.Take(), exp.Take())
	Add(b.Take(), a.Take())

	bit := b.Take()
	for eit := exp.Take(); !eit.Done(); eit.Next() {
		if *bit.Upk() != *eit.Upk() {
			t.Logf("test failed. exp: %v\n, got: %v\n", exp, b)
			t.Fail()
		}
		bit.Next()
	}
}

func TestSub(t *testing.T) {
	a := Reshape(Arange(0, 15), Shape{3, 5})
	b := Reshape(Arange(0, 15), Shape{3, 5})
	exp := Reshape(Arange(0, 15), Shape{3, 5})

	sub := func(x, y Iterator) {
		for !y.Done() {
			*y.Upk() -= *x.Upk()
			x.Next()
			y.Next()
		}
		x.Reset()
		y.Reset()
	}
	sub(a.Take(), exp.Take())
	Sub(b.Take(), a.Take())

	bit := b.Take()
	for eit := exp.Take(); !eit.Done(); eit.Next() {
		if *bit.Upk() != *eit.Upk() {
			t.Logf("test failed. exp: %v\n, got: %v\n", exp, b)
			t.Fail()
		}
		bit.Next()
	}
}

func TestView(t *testing.T) {
	a := Reshape(Arange(0, 60), Shape{3, 4, 5})
	b := a.View(Index{0, 1, 1}, Shape{3, 4})
	exp := []float64{6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19}
	got := make([]float64, 0, len(exp))

	for it := Iter(b); !it.Done(); it.Next() {
		got = append(got, *it.Upk())
	}

	for i, v := range exp {
		if got[i] != v {
			t.Logf("test failed. exp: %v\n, got: %v\n", exp, got)
			t.Fail()
		}
	}
}

func BenchmarkFunction1(bn *testing.B) {
	// z = a*x^2 + b*y + c
	fn := func(a float64, x Iterator, b float64, y Iterator, c float64, z Iterator) {
		_x, _y := 0.0, 0.0
		for !z.Done() {
			_x = *x.Upk()
			_y = *y.Upk()
			*z.Upk() += a*_x*_x + b*_y + c

			x.Next()
			y.Next()
			z.Next()
		}
		x.Reset()
		y.Reset()
		z.Reset()
	}

	a := rand.Float64()
	b := rand.Float64()
	c := rand.Float64()
	x := Rand(TestArrayShape).Take()
	y := Rand(TestArrayShape).Take()
	z := Rand(TestArrayShape).Take()
	bn.ResetTimer()
	bn.ReportAllocs()
	for i := 0; i < bn.N; i++ {
		fn(a, x, b, y, c, z)
	}
}

func BenchmarkAxpy(bn *testing.B) {
	// y += x
	// a := rand.Float64()
	a := 1.0
	x := Rand(TestArrayShape).Take()
	y := Rand(TestArrayShape).Take()
	bn.ResetTimer()
	bn.ReportAllocs()
	for i := 0; i < bn.N; i++ {
		Axpy(a, x, y)
	}
}

func BenchmarkFunction3(bn *testing.B) {
	a := rand.Float64()
	b := rand.Float64()
	c := rand.Float64()
	x := Rand(TestArrayShape).Take()
	y := Rand(TestArrayShape).Take()
	z := Rand(TestArrayShape).Take()
	bn.ResetTimer()
	bn.ReportAllocs()

	// z = a*x^2 + b*y + c
	// Shift(Plus(Scale(Sq(x),a), Scale(y,b)), c)
	for i := 0; i < bn.N; i++ {
		z = Shift(Add(Scale(Sq(x), a), Scale(y, b)), c)
	}
	_ = z.Len()
}

func BenchmarkAdd(bn *testing.B) {
	x := Rand(TestArrayShape).Take()
	y := Rand(TestArrayShape).Take()
	bn.ResetTimer()
	bn.ReportAllocs()

	// y += x
	for i := 0; i < bn.N; i++ {
		Add(y, x)
	}
}

func BenchmarkSq(b *testing.B) {
	x := Rand(TestArrayShape)
	it := x.Take()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		Sq(it)
	}
}
