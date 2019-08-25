package nd

import (
	"fmt"
	"testing"
)

func TestIterator_Get(t *testing.T) {
	a := Reshape(Arange(0, 60), Shape{3, 4, 5})
	fmt.Println("a: ", a)

	exp := a.Data()
	got := make([]float64, 0, len(exp))

	for it := Iter(a); !it.Done(); it.Next() {
		got = append(got, *it.Upk())
	}

	for i, v := range exp {
		if v != got[i] {
			t.Logf("test failed. exp: %v, got: %v\n", exp, got)
			t.Fail()
		}
	}

	b := a.View(Index{0, 1, 2}, Shape{2, 2, 3})
	fmt.Println("b: ", b)

	exp = []float64{7, 8, 9, 12, 13, 14, 27, 28, 29, 32, 33, 34}
	got = make([]float64, 0, len(exp))
	for it := Iter(b); !it.Done(); it.Next() {
		got = append(got, *it.Upk())
	}

	for i, v := range exp {
		if v != got[i] {
			t.Logf("test failed. exp: %v, got: %v\n", exp, got)
			t.Fail()
		}
	}
}

func TestNdarray_Take(t *testing.T) {
	array := Reshape(Arange(0, 12), Shape{3, 4})
	fmt.Println(array)
	// _ = array.String()
	// 5,7,9,11
	exp := []float64{5, 7, 9, 11}
	got := make([]float64, 0, len(exp))
	p := array.Take().From(5).To(11).WithStep(2)
	for ; !p.Done(); p.Next() {
		got = append(got, *p.Upk())
	}
	for i, v := range exp {
		if got[i] != v {
			t.Logf("test failed. exp: %v, got: %v\n", exp, got)
			t.Fail()
		}
	}

	exp = []float64{5, 6, 7, 8, 9, 10, 11}
	got = make([]float64, 0, len(exp))
	for p := array.Take().From(5).To(11); !p.Done(); p.Next() {
		got = append(got, *p.Upk())
	}
	for i, v := range exp {
		if got[i] != v {
			t.Logf("test failed. exp: %v, got: %v\n", exp, got)
			t.Fail()
		}
	}

	exp = []float64{5, 6, 7, 8, 9, 10, 11}
	got = make([]float64, 0, len(exp))
	for p := array.Take().From(5); !p.Done(); p.Next() {
		got = append(got, *p.Upk())
	}
	for i, v := range exp {
		if got[i] != v {
			t.Logf("test failed. exp: %v, got: %v\n", exp, got)
			t.Fail()
		}
	}
}

func TestIterator_Upk(t *testing.T) {
	a := Reshape(Arange(0, 15), Shape{1, 15})
	for it := a.Take().(*iterator); !it.Done(); it.Next() {
		v := it.Upk()
		*v *= 2
	}
	fmt.Println("a: ", a)
}

// func TestIterator_FromToStep(t *testing.T) {
// 	a := Reshape(Arange(0, 60), Shape{3, 4, 5})
// 	fmt.Println(a)
// 	it := a.Take()
// 	it.From(Sub2ind(a, Index{1, 0, 1})).To(Sub2ind(a, Index{1, 3, 1})).WithStep(a.Strides()[1])
//
// 	dot := func(n int, x, y []float64, incx, incy int) (s float64) {
// 		for ix, iy := 0, 0; n != 0; ix, iy = ix+incx, iy+incy {
// 			s += x[ix] * y[iy]
// 			n--
// 		}
// 		return
// 	}
//
// 	dot2d := func(shape, xstrides, ystrides Shape, x, y []float64, ind Index) (s float64) {
// 		for i := 0; i < shape[0]; i++ {
// 			ind[0] = i
// 			xb := sub2ind(xstrides, ind)
// 			yb := sub2ind(ystrides, ind)
// 			fmt.Println("xb, yb: ", shape, xstrides, ystrides)
// 			s += dot(shape[1], x[xb:], y[yb:], xstrides[1], ystrides[1])
// 		}
// 		return s
// 	}
//
// 	b := a.View(Index{0, 0, 0}, Shape{1, 4})
//
// 	ndot := func(x, y Array) (s float64) {
// 		if x.Ndims() != y.Ndims() {
// 			panic("rank mismatch")
// 		}
//
// 		if !isShapeSame(x, y) {
// 			panic("dimensions mismatch")
// 		}
//
// 		ndims := x.Ndims()
// 		if ndims < 3 {
// 			xstrides, ystrides := x.Strides(), y.Strides()
// 			return dot2d(x.Shape(), xstrides, ystrides, x.Data(), y.Data(), Index{0, 0})
// 		}
// 		shape := make(Shape, ndims)
// 		copy(shape, x.Shape()[:ndims-2])
// 		for i := ndims - 2; i < ndims; i++ {
// 			shape[i] = 1
// 		}
//
// 		ind := make(Index, ndims)
// 		xv := x.View(ind, shape)
// 		yv := y.View(ind, shape)
//
// 		return
// 	}
// 	v := ndot(b, b)
// 	fmt.Println(v)
//
// 	for ; !it.Done(); it.Next() {
// 		fmt.Println(it.I(), it.(*iterator).ind)
// 	}
// }

// Benchmarks

func BenchmarkIterator_Get(b *testing.B) {
	b.ReportAllocs()
	a := Rand(TestArrayShape)
	it := a.Take().(*iterator)
	v := 0.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for it.Reset(); !it.Done(); it.Next() {
			v = *it.Upk()
		}
	}
	_ = v * v
}

func BenchmarkIter(b *testing.B) {
	b.ReportAllocs()
	a := Rand(TestArrayShape)
	var it Iterator
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		it = Iter(a)
	}
	_ = it.Len()
}

func BenchmarkIterator_At(b *testing.B) {
	b.ReportAllocs()
	a := Rand(TestArrayShape)
	it := a.Take()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for k := 0; k < a.Size(); k++ {
			it.At(k)
		}
	}
}
