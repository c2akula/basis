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

func BenchmarkIterator_Get(b *testing.B) {
	b.ReportAllocs()
	a := Rand(Shape{4, 35, 15})
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

func BenchmarkIter(b *testing.B) {
	b.ReportAllocs()
	a := Rand(Shape{4, 35, 15})
	var it Iterator
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		it = Iter(a)
	}
	_ = it.Len()
}

func BenchmarkIterator_At(b *testing.B) {
	b.ReportAllocs()
	a := Rand(Shape{4, 35, 15})
	it := a.Take()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for k := 0; k < a.Size(); k++ {
			it.At(k)
		}
	}
}
