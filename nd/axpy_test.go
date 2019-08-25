package nd

import (
	"math/rand"
	"testing"
)

func TestAxpy(t *testing.T) {
	x := Reshape(Arange(0, 60), Shape{3, 4, 5})
	y := Reshape(Arange(0, 60), Shape{3, 4, 5})
	exp := Zeroslike(y)
	it := exp.Take()

	for ; !it.Done(); it.Next() {
		*it.Upk() = x.Get(it.I()) + y.Get(it.I())
	}
	it.Reset()

	Axpy(1, x, y)

	for ; !it.Done(); it.Next() {
		if *it.Upk() != y.Get(it.I()) {
			t.Logf("test 'Axpy' failed. exp: %v\n, got: %v\n", exp, y)
			t.Fail()
		}
	}
}

func TestAxpyView(t *testing.T) {
	xv := Reshape(Arange(0, 60), Shape{3, 4, 5}).View(Index{1, 0, 1}, Shape{2, 3})
	yv := Reshape(Arange(0, 60), Shape{3, 4, 5}).View(Index{1, 1, 1}, Shape{2, 3})
	ev := Zeroslike(yv)
	it := Iter(ev)

	for ; !it.Done(); it.Next() {
		*it.Upk() = xv.Get(it.I()) + yv.Get(it.I())
	}
	it.Reset()

	Axpy(1, xv, yv)

	for ; !it.Done(); it.Next() {
		if *it.Upk() != yv.Get(it.I()) {
			t.Logf("test 'Axpy' failed. exp: %v\n, got: %v\n", ev, yv)
			t.Fail()
		}
	}
}

func BenchmarkAxpy(bn *testing.B) {
	a := rand.Float64()
	x := Rand(TestArrayShape)
	y := Rand(TestArrayShape)
	bn.ResetTimer()
	bn.ReportAllocs()
	for i := 0; i < bn.N; i++ {
		Axpy(a, x, y)
	}
}
